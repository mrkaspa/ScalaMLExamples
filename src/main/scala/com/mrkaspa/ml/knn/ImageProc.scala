package com.mrkaspa.ml.knn

import java.io.File

import breeze.linalg._

import scala.io.Source

/**
 * Created by michelperez on 9/17/15.
 */
object ImageProc extends KNN with App {

  val K = 3

  def img2vector(file: File): VDouble = {
    println(s"***converting >> ${file.getPath}")
    val bigArray =
      Source
        .fromFile(file)
        .getLines()
        .toList
        .foldLeft(Array[Double]()) { (acc, line) =>
        acc ++ line.split("").map(_.toDouble)
      }
    DenseVector[Double](bigArray)
  }

  def trainingMat(path: String): (MDouble, Array[Double]) = {
    val dir = new File(getClass.getResource(path).getFile)
    val files = dir.listFiles()
    val labels = scala.collection.mutable.MutableList[Double]()
    val startMatrix = DenseMatrix.zeros[Double](files.size, 1024)
    files.zipWithIndex.foreach { case (file, i) =>
      labels += file.getName().split("_")(0).toDouble
      startMatrix(i, ::) := img2vector(file).t
    }
    (startMatrix, labels.toArray)
  }

  def testMat(mat: MDouble, labels: Array[Double], testPath: String): (Int, Double) = {

    def testKNN(file: File): Boolean = {
      val label = file.getName().split("_")(0).toInt
      val input = img2vector(file).asDenseMatrix
      classify(input, mat, labels, K) == label
    }

    val dir = new File(getClass.getResource(testPath).getFile)
    val files = dir.listFiles()
    val errCount = files.foldLeft(0) { (acc, file) =>
      if (testKNN(file)) acc else acc + 1
    }
    (errCount, errCount / files.size.toDouble)
  }

  val timestamp: Long = System.currentTimeMillis
  val path = "/digits/trainingDigits/"
  val testPath = "/digits/testDigits/"
  val (mat, labels) = trainingMat(path)
  val (errCount, accuracy) = testMat(mat, labels, testPath)
  println(s"errCount >> $errCount")
  println(s"accuracy >> $accuracy")
  val lastTimestamp: Long = System.currentTimeMillis
  println("elapsed time")
  println((lastTimestamp - timestamp) / 1000d)

}
