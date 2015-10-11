package com.mrkaspa.ml.naivebayes

import breeze.linalg._

/**
 * Created by michelperez on 10/9/15.
 */
object NaiveBayes extends App {

  def loadDataSet(): (List[List[String]], List[Int]) = {
    val postingList = List(
      List("my", "dog", "has", "flea", "problems", "help", "please"),
      List("maybe", "not", "take", "him", "to", "dog", "park", "stupid"),
      List("my", "dalmation", "is", "so", "cute", "I", "love", "him"),
      List("stop", "posting", "stupid", "worthless", "garbage"),
      List("mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"),
      List("quit", "buying", "worthless", "dog", "food", "stupid")
    )
    val classList = List(0, 1, 0, 1, 0, 1)
    (postingList, classList)
  }

  def createVocabList(dataSet: List[List[String]]): Set[String] =
    dataSet.foldLeft(Set[String]())(_ ++ _)

  def bagOfWords2VecMN(vocab: Set[String], input: List[String]): List[Int] =
    vocab.toList.map(word => input.count(_ == word))

  def setOfWords2Vec(vocab: Set[String], input: List[String]): List[Int] =
    vocab.toList.map(word => if (input.contains(word)) 1 else 0)

  def trainNB0(trainMatrix: DenseMatrix[Double], trainCategory: DenseVector[Double]): (DenseVector[Double], DenseVector[Double], Double) = {
    val numTrainDocs = trainMatrix.rows
    val numWords = trainMatrix.cols
    val pAbusive = sum(trainCategory) / numTrainDocs.toFloat
    var p0Num = DenseVector.ones[Double](numWords)
    var p1Num = DenseVector.ones[Double](numWords)
    var p0Denom = 2.0
    var p1Denom = 2.0
    (0 until numTrainDocs).foreach { i =>
      val iterVec = trainMatrix(i, ::).t
      if (trainCategory(i) == 1.0) {
        p1Num = p1Num + iterVec
        p1Denom += sum(iterVec)
      } else {
        p0Num = p0Num + iterVec
        p0Denom += sum(iterVec)
      }
    }
    val p1Vect = p1Num :/ p1Denom
    val p0Vect = p0Num :/ p0Denom
    (p0Vect, p1Vect, pAbusive)
  }

  def classifyNB(vec2Classify: DenseVector[Double], p0Vec: DenseVector[Double], p1Vec: DenseVector[Double], pClass1: Double): Int = {
    val p1 = sum(vec2Classify :* p1Vec) + scala.math.log(pClass1)
    val p0 = sum(vec2Classify :* p0Vec) + scala.math.log(1.0 - pClass1)
    if (p1 > p0) 1 else 0
  }

  val (listOfPosts, listOfClasses) = loadDataSet()
  val vocabList = createVocabList(listOfPosts)
  println(s"vocabList >> $vocabList")
  val trainOne = listOfPosts.map(setOfWords2Vec(vocabList, _))
  val trainData = trainOne.foldLeft(Array[Int]()) { (acc, list) =>
    val arrIter = list.toArray
    acc ++ arrIter
  }
  val trainMatrix = new DenseMatrix(listOfPosts.size, vocabList.size, trainData, 0, vocabList.size, isTranspose = true)
  val trainCategory = DenseVector(listOfClasses.toArray)
  val (p0V, p1V, pAb) = trainNB0(convert(trainMatrix, Double), convert(trainCategory, Double))
  println(s"p0V >> $p0V")
  println(s"p1V >> $p1V")
  println(s"pAb >> $pAb")

  var testEntry = List("love", "my", "dalmation")
  var thisDoc = setOfWords2Vec(vocabList, testEntry)
  var classifiedAs = classifyNB(convert(DenseVector(thisDoc.toArray), Double), p0V, p1V, pAb)
  println(s"classifiedAs >> $classifiedAs")

  testEntry = List("stupid", "garbage")
  thisDoc = setOfWords2Vec(vocabList, testEntry)
  classifiedAs = classifyNB(convert(DenseVector(thisDoc.toArray), Double), p0V, p1V, pAb)
  println(s"classifiedAs >> $classifiedAs")
}
