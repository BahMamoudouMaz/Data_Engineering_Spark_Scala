name := "proj"

version := "1.0"

scalaVersion := "2.12.17"  // Assurez-vous que la version de Scala correspond à celle utilisée par Spark

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.0",
  "org.apache.spark" %% "spark-sql" % "3.5.0",
  "org.apache.spark" %% "spark-mllib" % "3.5.0",
  "org.scalanlp" %% "breeze" % "2.0",
  "org.scalanlp" %% "breeze-viz" % "2.0"
)
