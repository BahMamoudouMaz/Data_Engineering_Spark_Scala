import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object Main {
  def main(args: Array[String]): Unit = {
    // Initialisation de la session Spark
    val spark = SparkSession.builder
      .appName("Analyse des Annonces Airbnb")
      .master("local[*]")
      .getOrCreate()

    // Chargement des données des annonces Airbnb
    val listingsDF = loadData("listings.csv", spark)
    displaySchemaAndSample(listingsDF)

    // Nettoyage des données
    val cleanedListingsDF = cleanData(listingsDF)

    // Traitement des données
    val avgPriceDF = calculateAvgPriceByNeighbourhood(cleanedListingsDF)
    showHighestAvgPriceNeighbourhood(avgPriceDF)

    val highAvailabilityDF = filterHighAvailability(cleanedListingsDF)
    showCountByNeighbourhood(highAvailabilityDF)

    val roomTypeDistributionDF = calculateRoomTypeDistribution(cleanedListingsDF)
    roomTypeDistributionDF.show()

    val avgReviewByNeighbourhoodDF = calculateAvgReviewByNeighbourhood(cleanedListingsDF)
    showTop10NeighbourhoodsByReview(avgReviewByNeighbourhoodDF)

    showMostFrequentPrice(cleanedListingsDF)
    val totalReviewsDF = calculateTotalReviews(cleanedListingsDF)
    showTop10MostReviewedListings(totalReviewsDF)

    val rankedListingsDF = rankListingsByPrice(cleanedListingsDF)
    showTopThreeExpensiveListings(rankedListingsDF)

    // Enregistrement des DataFrames dans des vues temporaires pour SQL
    registerTempViews(rankedListingsDF)

    // Requêtes SQL
    showLowestAvgPriceNeighbourhoods(spark)
    showListingsByRoomType(spark)

    // Chargement des données sur les incidents
    val crimeDF = loadData("crime_violence.csv", spark)
    val enrichedCrimeDF = enrichCrimeData(crimeDF, cleanedListingsDF)
    writeCombinedData(enrichedCrimeDF)

    spark.stop()
  }

  def loadData(filePath: String, spark: SparkSession): DataFrame = {
    spark.read
      .option("header", "true")
      .option("sep", ";")
      .csv(filePath)
  }

  def displaySchemaAndSample(df: DataFrame): Unit = {
    df.printSchema()
    df.show(5, false)
  }

  def cleanData(df: DataFrame): DataFrame = {
    df.na.drop("any", Seq("price", "number_of_reviews"))
      .withColumn("price", regexp_replace(col("price"), "\\$", "").cast("double"))
  }

  def calculateAvgPriceByNeighbourhood(df: DataFrame): DataFrame = {
    df.groupBy("neighbourhood", "room_type")
      .agg(avg("price").alias("average_price"))
  }

  def showHighestAvgPriceNeighbourhood(df: DataFrame): Unit = {
    df.orderBy(desc("average_price")).show()
  }

  def filterHighAvailability(df: DataFrame): DataFrame = {
    df.filter(col("availability_365") > 300)
  }

  def showCountByNeighbourhood(df: DataFrame): Unit = {
    df.groupBy("neighbourhood").count().show()
  }

  def calculateRoomTypeDistribution(df: DataFrame): DataFrame = {
    df.groupBy("neighbourhood", "room_type").count()
  }

  def calculateAvgReviewByNeighbourhood(df: DataFrame): DataFrame = {
    df.groupBy("neighbourhood")
      .agg(avg("reviews_per_month").alias("average_reviews_per_month"))
  }

  def showTop10NeighbourhoodsByReview(df: DataFrame): Unit = {
    df.orderBy(desc("average_reviews_per_month")).show(10, false)
  }

  def showMostFrequentPrice(df: DataFrame): Unit = {
    val mostFrequentPrice = df.groupBy("price")
      .agg(count("*").alias("frequency"))
      .orderBy(desc("frequency"))
      .limit(1)

    println("Most Frequent Price:")
    println("#######################################################################################")
    mostFrequentPrice.show()
    println("#######################################################################################")
  }

  def calculateTotalReviews(df: DataFrame): DataFrame = {
    df.groupBy("room_type")
      .agg(sum("number_of_reviews").alias("total_reviews"))
  }

  def showTop10MostReviewedListings(df: DataFrame): Unit = {
    df.orderBy(desc("total_reviews")).show(10, false)
  }

  def rankListingsByPrice(df: DataFrame): DataFrame = {
    val windowSpec = Window.partitionBy("neighbourhood").orderBy(desc("price"))
    df.withColumn("rank", rank().over(windowSpec))
  }

  def showTopThreeExpensiveListings(df: DataFrame): Unit = {
    df.filter(col("rank") <= 3).show()
  }

  def registerTempViews(df: DataFrame): Unit = {
    df.createOrReplaceTempView("RankedListings")
  }

  def showLowestAvgPriceNeighbourhoods(spark: SparkSession): Unit = {
    val lowestAvgPriceDF = spark.sql("""
        SELECT neighbourhood, AVG(price) as avg_price
        FROM RankedListings
        GROUP BY neighbourhood
        ORDER BY avg_price ASC
        LIMIT 5
    """)
    lowestAvgPriceDF.show()
  }

  def showListingsByRoomType(spark: SparkSession): Unit = {
    val listingsByRoomTypeDF = spark.sql("""
        SELECT neighbourhood, room_type, COUNT(*) as num_listings
        FROM RankedListings
        GROUP BY neighbourhood, room_type
        ORDER BY neighbourhood, room_type
    """)
    listingsByRoomTypeDF.show()
  }

  def enrichCrimeData(crimeDF: DataFrame, listingsDF: DataFrame): DataFrame = {
    val renamedCrimeDF = crimeDF.withColumnRenamed("Analysis Neighborhood", "neighbourhood")
    val incidentsByNeighbourhoodDF = renamedCrimeDF.groupBy("neighbourhood")
      .agg(count("*").alias("total_incidents"))

    val avgPriceByNeighbourhoodDF = listingsDF.groupBy("neighbourhood")
      .agg(avg("price").alias("avg_price"))

    avgPriceByNeighbourhoodDF.join(incidentsByNeighbourhoodDF, "neighbourhood", "inner")
  }

  def writeCombinedData(df: DataFrame): Unit = {
    df.write
      .option("header", "true")
      .csv("combined_neighbourhood_data.csv")
  }
}
