package com.mrkaspa.ml.demo

import breeze.linalg._

/**
 * Created by michelperez on 6/10/15.
 */
object Main extends App {

  val x: DenseVector[Int] = DenseVector.zeros[Int](5) + 1

  val y = DenseVector(1, 2, 3, 4, 5)

  //add
  println(x + y)

  //transpose and multiplication
  println(Transpose(x) * y)

  //slicing
  x(2 to 4) := 10
  println(x)

  //update a value
  x(0) = 11
  println(x)

  //Matrix
  val xx = DenseMatrix.zeros[Int](5, 5)
  val yy = DenseMatrix((1, 1), (2, 2), (3, 3), (4, 4), (5, 5))

  val inverse = inv(DenseMatrix((1,1), (2,3)))

  xx(::, 1) := DenseVector(1, 2, 3, 4, 5)

  xx(1, ::) := DenseVector(1, 2, 3, 4, 5).t

  println("inverse >> ")
  println(inverse)

  println("eye " + DenseMatrix.eye[Double](3))

  println(xx)

  println(xx * yy)

}
