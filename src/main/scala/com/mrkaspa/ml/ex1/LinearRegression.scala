package com.mrkaspa.ml.ex1

import java.io.File

import breeze.linalg._

import scala.annotation.tailrec

/**
 * Created by michelperez on 6/10/15.
 */
object LinearRegression extends App {

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

  //gets the ideal theta
  val (newTheta, _) = gradientDescent(X1, y, theta, alpha, iterations)

  println(s"newTheta = ${newTheta}")

  //predicts the value using the hypotesis θT * X
  val pred1 = DenseMatrix(1.0, 3.5).t * newTheta

  println(s"pred1 = $pred1")

  /**
   * The function to minimize theta
   * @param X the features matrix
   * @param y the results matrix
   * @param theta initial theta
   * @param alpha step value
   * @param iterations num of iterations for the algorithm
   * @return the minimized theta and the history for J the cost function
   * */
  def gradientDescent(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double], alpha: Double, iterations: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {

    //calculates the cost function
    def computeCost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double]): Double = {
      val m = y.rows
      sum((X * theta - y) :^ 2d) / (2 * m)
    }

    val m = y.rows

    //executes this function iter times
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
