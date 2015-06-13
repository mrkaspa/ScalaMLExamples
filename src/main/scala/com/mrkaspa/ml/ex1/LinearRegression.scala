package com.mrkaspa.ml.ex1

import java.io.File

import breeze.linalg._

import scala.annotation.tailrec

/**
 * Created by michelperez on 6/10/15.
 */
object LinearRegression extends App {

  val file = new File(getClass.getResource("/ex1data1.txt").getFile)
  val csv = csvread(file)
  val X = csv(::, 0).asDenseMatrix.t
  val y = csv(::, 1).asDenseMatrix.t
  val X1 = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X)
  val theta = DenseMatrix.zeros[Double](X1.cols, 1)
  val iterations = 1500
  val alpha = 0.01

  val (newTheta, _) = gradientDescent(X1, y, theta, alpha, iterations)

  println(s"newTheta = ${newTheta}")

  val pred1 = DenseMatrix(1.0, 3.5).t * newTheta

  println(s"pred1 = $pred1")

//  newTheta = -3.63029143940436
//  1.166362350335582
//  pred1 = 0.4519767867701767

  def gradientDescent(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double], alpha: Double, iterations: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {

    def computeCost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double]): Double = {
      val m = y.rows
      sum((X * theta - y) :^ 2d) / (2 * m)
    }

    val m = y.rows

    @tailrec
    def descent(newTheta: DenseMatrix[Double], history: DenseMatrix[Double], iter: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {
      iter match {
        case 0 => (newTheta, history)
        case _ =>
          val htheta: DenseMatrix[Double] = X1 * newTheta
          val theta0 = newTheta(0, 0) - alpha / m * sum((htheta - y) :* X(::, 0).t)
          val theta1 = newTheta(1, 0) - alpha / m * sum((htheta - y) :* X(::, 1).t)
          descent(
            DenseMatrix(theta0, theta1),
            DenseMatrix.horzcat(history, DenseMatrix(computeCost(X, y, theta)))
            , iter - 1
          )
      }
    }

    descent(theta, DenseMatrix(computeCost(X, y, theta)), iterations)

  }

}
