package org.apache.spark.mllib.regression

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, GradientDescent, Updater}
import org.apache.spark.rdd.RDD

/**
  * Created by zrf on 4/24/15.
  */

object FMWithSGD {
  /**
    * Train a Factoriaton Machine Regression model given an RDD of (label, features) pairs. We run a fixed number
    * of iterations of gradient descent using the specified step size. Each iteration uses
    * `miniBatchFraction` fraction of the data to calculate a stochastic gradient. The weights used
    * in gradient descent are initialized using the initial weights provided.
    *
    * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
    *              matrix A as well as the corresponding right hand side label y.
    * @param task 0 for Regression, and 1 for Binary Classification
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.
    * @param miniBatchFraction Fraction of data to be used per iteration.
    * @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
    *            one-way interactions should be used, and the number of factors that are used for pairwise
    *            interactions, respectively.
    * @param regParam A (Double,Double,Double) 3-Tuple stands for the regularization parameters of intercept, one-way
    *                 interactions and pairwise interactions, respectively.
    * @param initStd Standard Deviation used for factorization matrix initialization.
    */
  def train(input: RDD[LabeledPoint],
            task: Int,
            numIterations: Int,
            stepSize: Double,
            miniBatchFraction: Double,
            dim: (Boolean, Boolean, Int),
            regParam: (Double, Double, Double),
            initStd: Double,
            treeDepth: Int = 2): FMModel = {
    new FMWithSGD(task, stepSize, numIterations, dim, regParam, miniBatchFraction)
      .setInitStd(initStd)
      .setTreeDepth(treeDepth)
      .run(input)
  }

  def train(input: RDD[LabeledPoint],
            task: Int,
            numIterations: Int): FMModel = {
    new FMWithSGD(task, 1.0, numIterations, (true, true, 8), (0, 1e-3, 1e-4), 1e-5)
      .setInitStd(0.01)
      .run(input)
  }
}


class FMWithSGD(private var task: Int,
                private var stepSize: Double,
                private var numIterations: Int,
                private var dim: (Boolean, Boolean, Int),
                private var regParam: (Double, Double, Double),
                private var miniBatchFraction: Double) extends Serializable {


  /**
    * Construct an object with default parameters: {task: 0, stepSize: 1.0, numIterations: 100,
    * dim: (true, true, 8), regParam: (0, 0.01, 0.01), miniBatchFraction: 1.0}.
    */
  def this() = this(0, 1.0, 100, (true, true, 8), (0, 1e-3, 1e-4), 1e-5)

  private var k0: Boolean = dim._1
  private var k1: Boolean = dim._2
  private var k2: Int = dim._3

  private var r0: Double = regParam._1
  private var r1: Double = regParam._2
  private var r2: Double = regParam._3

  private var initMean: Double = 0
  private var initStd: Double = 0.01

  private var numFeatures: Int = -1
  private var minLabel: Double = Double.MaxValue
  private var maxLabel: Double = Double.MinValue

  private var treeDepth: Int = 2

  /**
    * A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the one-way
    * interactions should be used, and the number of factors that are used for pairwise interactions, respectively.
    */
  def setDim(dim: (Boolean, Boolean, Int)): this.type = {
    require(dim._3 > 0)
    this.k0 = dim._1
    this.k1 = dim._2
    this.k2 = dim._3
    this
  }

  /**
    *
    * @param addIntercept determines if the global bias term w0 should be used
    * @param add1Way determines if one-way interactions (bias terms for each variable)
    * @param numFactors the number of factors that are used for pairwise interactions
    */
  def setDim(addIntercept: Boolean = true, add1Way: Boolean = true, numFactors: Int = 8): this.type = {
    setDim((addIntercept, add1Way, numFactors))
  }


  /**
    * @param regParams A (Double,Double,Double) 3-Tuple stands for the regularization parameters of intercept, one-way
    *                  interactions and pairwise interactions, respectively.
    */
  def setRegParam(regParams: (Double, Double, Double)): this.type = {
    require(regParams._1 >= 0 && regParams._2 >= 0 && regParams._3 >= 0)
    this.r0 = regParams._1
    this.r1 = regParams._2
    this.r2 = regParams._3
    this
  }

  /**
    * @param regIntercept intercept regularization
    * @param reg1Way one-way interactions regularization
    * @param reg2Way pairwise interactions regularization
    */
  def setRegParam(regIntercept: Double = 0, reg1Way: Double = 0, reg2Way: Double = 0): this.type = {
    setRegParam((regIntercept, reg1Way, reg2Way))
  }


  /**
    * @param initStd Standard Deviation used for factorization matrix initialization.
    */
  def setInitStd(initStd: Double): this.type = {
    require(initStd > 0)
    this.initStd = initStd
    this
  }

  /**
    * Set fraction of data to be used for each SGD iteration.
    */
  def setMiniBatchFraction(miniBatchFraction: Double): this.type = {
    require(miniBatchFraction > 0 && miniBatchFraction <= 1)
    this.miniBatchFraction = miniBatchFraction
    this
  }

  /**
    * Set the number of iterations for SGD.
    */
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations > 0)
    this.numIterations = numIterations
    this
  }

  /**
    * Set the initial step size of SGD for the first step.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(stepSize: Double): this.type = {
    require(stepSize >= 0)
    this.stepSize = stepSize
    this
  }

  /**
   * Set the tree depth for aggregation
   * @param depth
   * @return
   */
  def setTreeDepth(depth: Int): this.type = {
    require(treeDepth >= 2)
    this.treeDepth = depth
    this
  }


  /**
    * Encode the FMModel to a dense vector, with its first numFeatures * numFactors elements representing the
    * factorization matrix v, sequential numFeaturs elements representing the one-way interactions weights w if k1 is
    * set to true, and the last element representing the intercept w0 if k0 is set to true.
    * The factorization matrix v is initialized by Gaussinan(0, initStd).
    * v : numFeatures * numFactors + w : [numFeatures] + w0 : [1]
    */
  private def generateInitWeights(): Vector = {
    (k0, k1) match {
      case (true, true) =>
        Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean) ++
          Array.fill(numFeatures + 1)(0.0))

      case (true, false) =>
        Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean) ++
          Array(0.0))

      case (false, true) =>
        Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean) ++
          Array.fill(numFeatures)(0.0))

      case (false, false) =>
        Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean))
    }
  }


  /**
    * Create a FMModle from an encoded vector.
    */
  private def createModel(weights: Vector): FMModel = {

    val values = weights.toArray

    val v = new DenseMatrix(k2, numFeatures, values.slice(0, numFeatures * k2))

    val w = if (k1) Some(Vectors.dense(values.slice(numFeatures * k2, numFeatures * k2 + numFeatures))) else None

    val w0 = if (k0) values.last else 0.0

    new FMModel(task, v, w, w0, minLabel, maxLabel)
  }


  /**
    * Run the algorithm with the configured parameters on an input RDD
    * of LabeledPoint entries.
    */
  def run(input: RDD[LabeledPoint]): FMModel = {

    this.numFeatures = input.first().features.size
    require(numFeatures > 0)

    if (task == 0) {
      val (minT, maxT) = input.map(_.label).aggregate[(Double, Double)]((Double.MaxValue, Double.MinValue))({
        case ((min, max), v) =>
          (Math.min(min, v), Math.max(max, v))
      }, {
        case ((min1, max1), (min2, max2)) =>
          (Math.min(min1, min2), Math.max(max1, max2))
      })

      this.minLabel = minT
      this.maxLabel = maxT
    }

    val gradient = new FMGradient(task, k0, k1, k2, numFeatures, minLabel, maxLabel)

    val updater = new FMUpdater(k0, k1, k2, r0, r1, r2, numFeatures)

    /*
    val optimizer = new GradientDescent(gradient, updater)
      .setStepSize(stepSize)
      .setNumIterations(numIterations)
      .setMiniBatchFraction(miniBatchFraction)
      .setConvergenceTol(Double.MinPositiveValue)
    */
    val data = task match {
      case 0 =>
        input.map(l => (l.label, l.features)).persist()
      case 1 =>
        input.map(l => (if (l.label > 0) 1.0 else -1.0, l.features)).persist()
    }

    val initWeights = generateInitWeights()

    val (weights, _) = MyGradientDescent.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      0.0,
      1.0,
      initWeights,
      Double.MinPositiveValue,
      2)

    createModel(weights)
  }
}


object MyGradientDescent extends Logging {
  /**
   * Run stochastic gradient descent (SGD) in parallel using mini batches.
   * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
   * in order to compute a gradient estimate.
   * Sampling, and averaging the subgradients over this subset is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data Input data for SGD. RDD of the set of data examples, each of
   *             the form (label, [feature values]).
   * @param gradient Gradient object (used to compute the gradient of the loss function of
   *                 one single data example)
   * @param updater Updater function to actually perform a gradient step in a given direction.
   * @param stepSize initial step size for the first step
   * @param numIterations number of iterations that SGD should be run.
   * @param regParam regularization parameter
   * @param miniBatchFraction fraction of the input data set that should be used for
   *                          one iteration of SGD. Default value 1.0.
   * @param convergenceTol Minibatch iteration will end before numIterations if the relative
   *                       difference between the current weight and the previous weight is less
   *                       than this value. In measuring convergence, L2 norm is calculated.
   *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */
  def runMiniBatchSGD(
    data: RDD[(Double, Vector)],
    gradient: Gradient,
    updater: Updater,
    stepSize: Double,
    numIterations: Int,
    regParam: Double,
    miniBatchFraction: Double,
    initialWeights: Vector,
    convergenceTol: Double,
    treeDepth: Int = 2): (Vector, Array[Double]) = {
    import breeze.linalg.{DenseVector => BDV}

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
     * For the first iteration, the regVal will be initialized as sum of weight squares
     * if it's L2 updater; for L1 updater, the same logic is followed.
     */
    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (gradientSum, lossSum, miniBatchSize, bcTime) = data.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L, 0L))(
          seqOp = (c, v) => {
            // c: (grad, loss, count, t), v: (label, features)
            val s = System.currentTimeMillis()
            val w = bcWeights.value
            val t = System.currentTimeMillis() - s
            val l = gradient.compute(v._2, v._1, w, Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1, c._4 + t)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count, t)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3, c1._4 + c2._4)
          }, treeDepth)

      logWarning(s"BC read time: $bcTime ms")
      if (miniBatchSize > 0) {
        /**
         * lossSum is computed using the weights from the previous iteration
         * and regVal is the regularization value computed in the previous iteration as well.
         */
        stochasticLossHistory.append(lossSum / miniBatchSize + regVal)
        val update = updater.compute(
          weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
          stepSize, i, regParam)
        weights = update._1
        regVal = update._2

        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights != None && currentWeights != None) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)

  }

  /**
   * Alias of [[runMiniBatchSGD]] with convergenceTol set to default value of 0.001.
   */
  def runMiniBatchSGD(
    data: RDD[(Double, Vector)],
    gradient: Gradient,
    updater: Updater,
    stepSize: Double,
    numIterations: Int,
    regParam: Double,
    miniBatchFraction: Double,
    initialWeights: Vector): (Vector, Array[Double]) =
  GradientDescent.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
    regParam, miniBatchFraction, initialWeights, 0.001)


  private def isConverged(
    previousWeights: Vector,
    currentWeights: Vector,
    convergenceTol: Double): Boolean = {
    import breeze.linalg.norm
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }

}