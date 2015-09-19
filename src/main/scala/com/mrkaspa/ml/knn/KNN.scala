package com.mrkaspa.ml.knn

import java.io.File

import breeze.linalg._

/**
 * Created by michelperez on 9/16/15.
 */
trait KNN {
  type MDouble = DenseMatrix[Double]
  type VDouble = DenseVector[Double]

  def meassure(s: String)(f: () => Unit): Unit = {
    val timestamp: Long = System.currentTimeMillis
    f()
    val lastTimestamp: Long = System.currentTimeMillis
    println(s"elapsed time in $s")
    println((lastTimestamp - timestamp) / 1000d)
  }

  def classify(input: MDouble, dataSet: MDouble, labels: Array[Double], k: Int): Int = {
    val dataSetSize = dataSet.rows
//    meassure("tile(input, dataSetSize, 1") { () => tile(input, dataSetSize, 1) }
    val tiled: MDouble = tile(input, dataSetSize, 1) // tiled data set
    // calculates the distance => √(x2-x1)^ + (y2-y1)^
//    meassure("tiled - dataSet") { () => tiled - dataSet }
    val diffMat: MDouble = tiled - dataSet
    var timestamp: Long = System.currentTimeMillis
//    meassure("diffMat :^ 2.0") { () => diffMat :^ 2.0 }
    val sqDiffMat: MDouble = diffMat :^ 2.0
//    meassure("sum(sqDiffMat(*, ::))") { () => sum(sqDiffMat(*, ::)) }
    val sqDistances: VDouble = sum(sqDiffMat(*, ::)) // sum per row
//    meassure("sqDistances :^ 0.5") { () => sqDistances :^ 0.5 }
    val distances: VDouble = sqDistances :^ 0.5 // sqrt
//    meassure("argsort(distances)") { () => argsort(distances) }
    val sortedDistances: Seq[Int] = argsort(distances) // sort the arguments
//    meassure("mapa 1"){ () =>
//      (0 to k).map { i => labels(sortedDistances(i)) }.groupBy(identity).mapValues(_.size)
//    }
    val mapAcc = (0 to k).map { i => labels(sortedDistances(i)) }.groupBy(identity).mapValues(_.size)
//    meassure("mapa 2") { () =>
//      val res = mapAcc.foldLeft((0.0, 0)) { (t, curr) => if (t._2 > curr._2) t else curr }
//    }
    val res = mapAcc.foldLeft((0.0, 0)) { (t, curr) => if (t._2 > curr._2) t else curr }

    res._1.toInt
  }
}

object KNN extends KNN with App {
  val file = new File(getClass.getResource("/datingData.txt").getFile)
  val csv = csvread(file, separator = '\t')
  val mat = csv(::, 0 to 2)
  val labels = csv(::, 3).toArray
  val labelsMap = Map(3 -> "High", 2 -> "Medium", 1 -> "Low")

  val input: MDouble = DenseMatrix(Array(1000.0, 2.0, 5.0))

  val res = classify(input, mat, labels, 5)
  println(s"solucion => ${labelsMap(res)}")

}
