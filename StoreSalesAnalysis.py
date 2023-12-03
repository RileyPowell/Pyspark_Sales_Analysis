from pyspark.sql import SparkSession
from pyspark.sql.functions import lag
import pyspark
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


spark = SparkSession.builder.master("local[1]").appName("StoreAnalysis").getOrCreate()
def df_creation(file_name,form):
    data = spark.read.option("header",True).format(form).load(file_name)
    return data
train_df = df_creation('store-sales-time-series-forecasting/train.csv','csv')
train_df = train_df.withColumn("sales",train_df.sales.cast('int'))
train_df = train_df.withColumn("date",pyspark.sql.functions.col("date").cast(pyspark.sql.types.DateType()))
train_df = train_df.withColumn("day_of_week",pyspark.sql.functions.dayofweek(train_df['date']))
train_df = train_df.withColumn("day_of_year",pyspark.sql.functions.dayofyear(train_df['date']))
train_df = train_df.withColumn("week of year",pyspark.sql.functions.weekofyear(train_df['date']))
train_df = train_df.withColumn("year",pyspark.sql.functions.year(train_df['date']))

train_df.printSchema()


# Examine the performance of the stores over the previous years and how that performance has changed.
new_data = train_df.groupBy(["store_nbr","year"]).sum("sales").orderBy(['store_nbr',"year"])
new_data = new_data.withColumn("sales",new_data['sum(sales)'].cast("double"))
window = pyspark.sql.Window.partitionBy(["store_nbr"]).orderBy(["store_nbr","year"])
new_data = new_data.withColumn("previous_sales",lag("sum(sales)",1,0).over(window))
new_data = new_data.withColumn("sales",new_data['sales'].cast("double"))
new_data = new_data.withColumn("percent_change",pyspark.sql.functions.round((new_data["sales"]-new_data["previous_sales"])*100/new_data['previous_sales'],2))
new_data.show()








# Create lag features to capture current sales dependence on past sales
w = pyspark.sql.Window.partitionBy(["store_nbr","family"]).orderBy("date")
train_df = train_df.withColumn("lag_sales_1",lag("sales",1,0.0).over(w))
train_df = train_df.withColumn("lag_sales_2",lag("sales",2,0.0).over(w))
train_df = train_df.withColumn("lag_sales_3",lag("sales",3,0.0).over(w))


# Create columns showing the day of the week, week of the year, day of the year



# the window function should be written better because using rowsBetween assumes that there is a row for every date or
# that date is continuous, should change to range between.
rolling_avg_w = pyspark.sql.Window.partitionBy(['store_nbr','family']).orderBy("date").rowsBetween(-7,0)
train_df = train_df.withColumn("rolling_7_day_avg",pyspark.sql.functions.avg("sales").over(rolling_avg_w))
rolling_avg_w_month = pyspark.sql.Window.partitionBy(['store_nbr','family']).orderBy("date").rowsBetween(-30,0)
train_df = train_df.withColumn("rolling_30_day_avg",pyspark.sql.functions.avg("sales").over(rolling_avg_w_month))

train_df.show()






# Visualize the data distribution along with the trend generated from the rolling average.
train_df.createOrReplaceTempView("training_data")
pd_df = spark.sql("select * from training_data where family = 'AUTOMOTIVE' and store_nbr = 1").toPandas()
fig, ax = plt.subplots(figsize = (8,6))
year_month_formatter = mdates.DateFormatter("%Y-%m")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(year_month_formatter)
line1 = ax.scatter(pd_df["date"][:200],pd_df["sales"][:200],label="Automotive sales data from store #1")
line2 = ax.plot(pd_df["date"][:200],pd_df["rolling_7_day_avg"][:200],label="7 day rolling average of sales data")
line3 = ax.plot(pd_df["date"][:200],pd_df["rolling_30_day_avg"][:200],label="30 day rolling average of sales data")
ax.legend()
plt.show()








