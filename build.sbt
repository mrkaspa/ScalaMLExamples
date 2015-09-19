name := "MLCoursera"

version := "1.0"

scalaVersion := "2.11.6"

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

//val breezeVersion = "0.11.2"
val breezeVersion = "0.12-SNAPSHOT"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % breezeVersion,
  "org.scalanlp" %% "breeze-viz" % breezeVersion,
  "org.scalanlp" %% "breeze-natives" % breezeVersion
)