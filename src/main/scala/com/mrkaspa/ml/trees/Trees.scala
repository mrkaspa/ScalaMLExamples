package com.mrkaspa.ml.trees

import scala.collection.immutable.TreeMap


/**
 * Created by michelperez on 10/4/15.
 */
object Trees extends App {

  type Data = Array[Array[Int]]

  case class Tree[A, B](node: A, childs: Option[TreeMap[B, Tree[A, B]]])

  type TreeAny = Tree[Any, Int]

  val lnOf2 = scala.math.log(2)

  def createDataSet(): (Data, Array[String]) = {
    val data = Array(
      Array(1, 1, 1),
      Array(1, 1, 1),
      Array(1, 0, 0),
      Array(0, 1, 0),
      Array(0, 1, 0)
    )
    val labels = Array("no surfacing", "flippers")
    (data, labels)
  }

  def calcShannonEnt(data: Data): Double = {
    val grouped = data.map(_.last).groupBy(identity).mapValues(_.size)
    grouped.foldLeft(0.0) { (shannonEnt, labelPair) =>
      val prob = labelPair match {
        case (key, count) => count.toFloat / data.length
      }
      shannonEnt - (prob * scala.math.log(prob) / lnOf2)
    }
  }

  def splitDataSet(data: Data, axis: Int, value: Int): Data = {
    data.filter(_ (axis) == value).map(_.splitAt(axis) match {
      case (fl, fr) => fl ++ fr.tail
    })
  }

  def chooseBestToSplit(data: Data): Int = {
    val numFeatures = data(0).size - 1
    val baseEntropy = calcShannonEnt(data)
    val (_, result) = (0 until numFeatures).foldLeft((0.0, -1)) { (res, iter) =>
      val featList = for {row <- data} yield row(iter)
      val uniqueVals = featList.toSet
      val newEntropy = uniqueVals.foldLeft(0.0) { (acc, value) =>
        val subDataSet = splitDataSet(data, iter, value)
        val prob = subDataSet.size / data.size.toFloat
        acc + prob * calcShannonEnt(subDataSet)
      }
      val infoGain = baseEntropy - newEntropy
      res match {
        case (bestInfoGain, bestFeature) =>
          if (infoGain > bestInfoGain) {
            (infoGain, iter)
          } else {
            (bestInfoGain, bestFeature)
          }
      }
    }
    result
  }

  def majorityCount(arr: Array[Int]): Int = {
    arr.groupBy(identity).mapValues(_.size).foldLeft(0) { (max, curr) =>
      val (value, size) = curr
      if (size > max) value else max
    }
  }

  def createTree(data: Data, labels: Array[String]): TreeAny = {
    val classList = data.map(_.last)
    if (classList.forall(_ == classList(0))) return Tree(classList(0), None)
    if (data(0).size == 1) return Tree(majorityCount(classList), None)
    val bestFeat = chooseBestToSplit(data)
    val bestFeatLabel = labels(bestFeat)
    val newLabels = labels.splitAt(bestFeat) match {
      case (fl, fr) => fl ++ fr.tail
    }
    val featValues = data.map(_ (bestFeat))
    val uniqueVals = featValues.toSet
    val childs = uniqueVals.foldLeft(TreeMap[Int, TreeAny]()) { (tree, value) =>
      tree + (value -> createTree(splitDataSet(data, bestFeat, value), newLabels))
    }
    Tree(bestFeatLabel, Some(childs))
  }

  def printTree[A, B: Ordering](tree: Tree[A, B]): Unit = {

    def groupByLevels[A, B: Ordering](tree: Tree[A, B], level: Int, map: TreeMap[Int, List[A]]): TreeMap[Int, List[A]] = {
      val mapKey = if (!map.contains(level)) {
        map + (level -> List())
      }
      else {
        map
      }
      val mapValue = mapKey + (level -> (tree.node +: mapKey(level)))
      tree.childs.getOrElse(TreeMap[B, Tree[A, B]]()).foldLeft(mapValue) { (treeMap, childTree) =>
        val (key, valTree) = childTree
        treeMap ++ groupByLevels(valTree, level + 1, treeMap)
      }
    }

    val printable = groupByLevels(tree, 0, TreeMap())

    printable.foreach { case (key, list) =>
      println(s"[${key}] >> ${list}")
    }
  }

  val (data, labels) = createDataSet()
  println(s"Entropy >> ${calcShannonEnt(data)}")

  val splitted = splitDataSet(data, 0, 0)

  val bestToSplit = chooseBestToSplit(data)

  println(s"Best >> $bestToSplit")

  println("Solution >>")

  val tree = createTree(data, labels)
  println(tree)

  printTree(tree)

}
