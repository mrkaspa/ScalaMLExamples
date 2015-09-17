package com.mrkaspa.ml.linearRegresion

import java.io.File

import breeze.linalg._

import scala.annotation.tailrec

/**
 * Created by michelperez on 6/10/15.
 */
object LinearRegressionNormalEquation extends App {

  //read the file
  val file = new File(getClass.getResource("/ex1data1.txt").getFile)
  val csv = csvread(file)
  val X = csv(::, 0).asDenseMatrix.t
  val y = csv(::, 1).asDenseMatrix.t

  //concatenates a col [1...1] to the features matrix
  val X1 = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X)

  //theta initialized
  val theta = DenseMatrix.zeros[Double](X1.cols, 1)
  val iterations = 1500
  val alpha = 0.01

  //gets the ideal theta using normal equation solution
  val XX: DenseMatrix[Double] = (X1.t * X1)
  val newTheta = pinv(XX) * X1.t * y

  println(s"newTheta = ${newTheta}")

  //predicts the value using the hypotesis θT * X
  val pred1 = DenseMatrix(1.0, 3.5).t * newTheta

  println(s"pred1 = $pred1")

}
