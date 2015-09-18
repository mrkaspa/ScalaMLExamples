name := "MLCoursera"

version := "1.0"

scalaVersion := "2.11.6"

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

val breezeVersion = "0.11.2"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % breezeVersion,
  "org.scalanlp" %% "breeze-viz" % breezeVersion,
  "org.scalanlp" %% "breeze-natives" % breezeVersion,
)