package com.mrkaspa.ml.ex1

import java.io.File

import breeze.linalg._
import breeze.optimize._

/**
 * Created by michelperez on 6/13/15.
 */
object LinearRegressionWithMinimizer extends App {

  val file = new File(getClass.getResource("/ex1data1.txt").getFile)
  val csv = csvread(file)
  val X = csv(::, 0).asDenseMatrix.t
  val y = csv(::, 1).asDenseMatrix.t
  val X1 = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X)

  val f = new DiffFunction[DenseVector[Double]] {
    //calculates J and theta
    def calculate(theta: DenseVector[Double]) = {
      val m = y.rows
      val h = X1 * theta.asDenseMatrix.t
      val sqrErrors = (h - y) :^ 2d
      val J = 1d / (2d * m) * sum(sqrErrors)
      val grad: DenseMatrix[Double] = (1d / m) * (X1.t * (h - y))
      (J, grad.toDenseVector)
    }
  }

  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 1000, m = 5)

  val initialTheta = DenseVector(0.0, 0.0)

  //minimize theta using the LBFGS algorithm
  val newTheta = lbfgs.minimize(f, initialTheta)

  println(s"newTheta >> $newTheta")

  val pred1 = DenseMatrix(1.0, 3.5).t * newTheta

  println(s"pred1 = $pred1")


}
