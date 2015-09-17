package com.mrkaspa.ml.knn

import java.io.File

import breeze.linalg._

/**
 * Created by michelperez on 9/16/15.
 */
object KNN extends App {

  type MDouble = DenseMatrix[Double]
  type VDouble = DenseVector[Double]

  def classify(input: MDouble, dataSet: MDouble, labels: Array[Double], k: Int): Int = {
    val dataSetSize = dataSet.rows
    val tiled: MDouble = tile(input, dataSetSize, 1)
    val diffMat: MDouble = tiled - dataSet
    val sqDiffMat: MDouble = diffMat :^ 2.0
    val sqDistances: VDouble = sum(sqDiffMat(*, ::)) // sume las filas
    val distances: VDouble = sqDistances :^ 0.5
    val sortedDistances: Seq[Int] = argsort(distances)
    val mapAcc = (0 to k).map { i => labels(sortedDistances(i)) }.groupBy(identity).mapValues(_.size)
    val res = mapAcc.foldLeft((0.0, 0)) { (t, curr) => if (t._2 > curr._2) t else curr }
    res._1.toInt
  }

  val file = new File(getClass.getResource("/datingData.txt").getFile)
  val csv = csvread(file, separator = '\t')
  val mat = csv(::, 0 to 2)
  val labels = csv(::, 3).toArray
  val labelsMap = Map(3 -> "High", 2 -> "Medium", 1 -> "Low")

  val input: MDouble = DenseMatrix(Array(1000.0, 2.0, 5.0))

  val res = classify(input, mat, labels, 5)
  println(s"solucion => ${labelsMap(res)}")

}
