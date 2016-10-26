
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Row, SparkSession}



object TestFM extends App {

  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("test-fm")
      .master("local[*]")
      .getOrCreate()

    val inputPath = "PATH_TO_DATA/rcv1_train.binary"
    val data = spark.read.format("libsvm").load(inputPath)
      .rdd.map { case Row(l: Double, v: org.apache.spark.ml.linalg.Vector) =>
      LabeledPoint(l, Vectors.fromML(v))
    }.cache()

    val splits = data.randomSplit(Array(0.8, 0.2))
    val (train, test) = (splits(0), splits(1))

    //    val task = args(1).toInt
    //    val numIterations = args(2).toInt
    //    val stepSize = args(3).toDouble
    //    val miniBatchFraction = args(4).toDouble

    val fm1 = FMWithSGD.train(train, task = 1, numIterations = 100, stepSize = 1.0, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0.1, 0.1), initStd = 0.1)
    val preds1 = fm1.predict(test.map(_.features)).zip(test.map(_.label))
    val auc1 = new BinaryClassificationMetrics(preds1).areaUnderROC()
    val fm2 = FMWithLBFGS.train(train, task = 1, numIterations = 20, numCorrections = 5, dim = (false, true, 4), regParam = (0, 0.1, 0.1), initStd = 0.1)
    val preds2 = fm2.predict(test.map(_.features)).zip(test.map(_.label))
    val auc2 = new BinaryClassificationMetrics(preds2).areaUnderROC()

    println(auc1)
    println(auc2)
    println(fm1.factorMatrix)
    println(fm2.factorMatrix)
  }
}
