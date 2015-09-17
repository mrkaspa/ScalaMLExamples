package com.mrkaspa.ml.logisticRegresion

import java.io.File

import breeze.linalg._
import breeze.numerics.log
import breeze.optimize.{LBFGS, DiffFunction}

/**
 * Created by michelperez on 6/15/15.
 */
object LogisticRegression extends App {

  val file = new File(getClass.getResource("/ex2data1.txt").getFile)
  val csv = csvread(file)
  val X = csv(::, 0 to 1)
  val y = csv(::, 2).asDenseMatrix.t

  //concatenates a col [1...1] to the features matrix
  val X1 = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X)

  //function used in logistic regression
  def sigmoid(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    m.map { value =>
      1d / (1d + scala.math.exp(-value))
    }
  }

  val f = new DiffFunction[DenseVector[Double]] {

    //calculates J and theta
    def calculate(theta: DenseVector[Double]) = {
      val m = y.rows
      //the hypotesis here changes for logistic regression
      val h = sigmoid(X1 * theta.asDenseMatrix.t)
      val sqrErrors = (h - y) :^ 2d
      val costPos = (-y.t) * log(h)
      val costNeg = (1d - y.t) * log(1d - h)
      val Jm = 1d / m * (costPos - costNeg)
      val grad: DenseMatrix[Double] = (1d / m) * (X1.t * (h - y))
      val J = Jm(0, 0)
      (J, grad.toDenseVector)
    }
  }

  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 1000, m = 5)

  val initialTheta = DenseVector(0.0, 0.0, 0.0)

  println()

  //minimize theta using the LBFGS algorithm
  val newTheta = lbfgs.minimize(f, initialTheta)

  println(s"newTheta >> $newTheta")

  val prob = sigmoid((DenseMatrix(1.0, 45.0, 85.0).t * newTheta).asDenseMatrix);

  println(s"prob >> $prob")

}
