### -------- Install Required Packages -------- ###
install.packages("tidyverse")

### -------- Load Required Packages -------- ###
library(tidyverse) # For data storage, manipulation and visualization

### -------- Load Data Frames -------- ###
# This code will work provided the data is in a folder called f1_raw_data
# within the working directory

folder <- "./f1_raw_data"  # Define folder name

raw_f1_files <- list.files(folder, pattern="*.csv")  # List of file names

# Remove '.csv' from each string in list, convert all to lower case
raw_f1_files <- raw_f1_files %>%
  str_remove_all(".csv") %>%
  tolower()

# Load each csv file mapped from raw_f1_files into individual variables
for(i in raw_f1_files){
  filepath <- file.path(folder,paste(i,".csv", sep=""))
  assign(i, read.csv(filepath))
}

rm(filepath, folder, i) # Remove redundant variables

### -------- Initial DF Exploration -------- ###

# Explore each of the datasets
map(list(mget(raw_f1_files)), str)

# Memory of Data (Mega Bytes)
sort(sapply(ls(),function(x){format(object.size(get(x)), units="Mb")}))

### -------- DATA CLEANSING -------- ###

### -------- Remove Unnecessary Columns -------- ###

circuits <- circuits %>% select(circuitId, circuitRef, location, country, lat, lng)
constructors <- constructors %>% select(constructorId, name, nationality)
drivers <- drivers %>% select(driverId, driverRef, forename, surname, dob, nationality)
races <- races %>% select(raceId, year, date, round, circuitId)
results <- results %>% select(resultId, raceId, driverId, constructorId, grid, position, positionOrder,
                   points, laps, milliseconds, fastestLap, fastestLapSpeed, statusId)

# Setting Date column to character for efficient null detection in loop
drivers$dob <- as.character(drivers$dob)
races$date <- as.character(races$date)

# Add data.frames to list for looping
df_list <- list(circuits, constructors, drivers, races, results, status)

### -------- Count nulls in all dataframes -------- ###

# Count nulls function
count_nulls <- function(dataframe){
  apply(X = dataframe=="\\N", MARGIN = 2, FUN = sum) # Return count of \N values in each column
}

# Apply function to all dataframes
lapply(df_list, count_nulls)

### -------- Handle Nulls in Results table -------- ###
 
results[results == "\\N"] <- NA
head(results, 10)

### -------- Handling type casting for each Data.frame -------- ###
### -------- and renaming columns -------- ###

circuits <- circuits %>% mutate_at(c('circuitId'), as.integer) %>%
  mutate_at(c('circuitRef', 'location', 'country'), as.factor)


constructors <- constructors %>% mutate_at(c('constructorId'), as.integer) %>%
  mutate_at(c('name', 'nationality'), as.factor) %>%
  rename(constructor_name = name, constructor_nationality = nationality)


drivers <- drivers %>% mutate_at(c('driverId'), as.integer) %>%
  mutate_at(c('driverRef', 'nationality'), as.factor) %>%
  mutate_at(c('dob'), as.Date) %>%
  rename(driver_dob = dob, driver_nationality = nationality)


races <- races %>% mutate_at(c('raceId', 'year', 'round', 'circuitId'), as.integer) %>% 
  mutate_at(c('date'), as.Date)


results <- results %>% mutate_at(c('resultId', 'raceId', 'driverId', 'constructorId',
                                   'grid', 'position', 'positionOrder', 'laps', 'milliseconds',
                                   'fastestLap', 'statusId'), as.integer) %>%
  mutate_at(c('fastestLapSpeed'), as.double)

# Check max number in milliseconds (Less than 2*10^9, so can be left as integer)
max(results$milliseconds, na.rm = TRUE)

status <- status %>% mutate_at(c('statusId'), as.integer) %>% mutate_at(c('status'), as.factor)

# Use str on all datasets again to check type casting and column renaming worked
map(list(mget(raw_f1_files)), str)

# Create master dataframe, joining each table to results table
master_results <- results %>%
  select(-c(resultId)) %>%
  left_join(races, by = "raceId") %>%
  left_join(circuits, by = "circuitId") %>%
  left_join(drivers, by = "driverId") %>%
  left_join(constructors, by = "constructorId") %>%
  left_join(status, by = "statusId")

### -------- Calculated Columns -------- ###
# Calculate Driver Age Manually
master_results$driver_age <- as.numeric(master_results$date - master_results$driver_dob) %/% 365.25

# New column for Driver full name
master_results$driver_fullname <- str_c(master_results$forename, ' ', master_results$surname)

# Type casting new columns
master_results <- master_results %>% mutate_at(c('driver_fullname'), as.factor) %>% 
  mutate_at(c('driver_age'), as.integer)


### ~~~~~~~~~~~~~~~~~~~~~~~~~ CLEANSED DATSET ~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Reorder columns
master_results <- master_results %>%
  select(raceId, date, year, round, grid, position, positionOrder, points, laps, milliseconds,
         fastestLap, fastestLapSpeed, statusId, status, driverRef, driver_fullname, driver_nationality,
         driver_age, constructor_name, constructor_nationality, circuitRef, location, country, lat, lng)

# Order by date and position order to make results more readable
master_results <- master_results[order(master_results$date, master_results$positionOrder),]


write.csv(master_results, "formula_1_cleansed.csv")
str(master_results)
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

### group_by driver and create summary stats ### 
master_results %>%
  group_by(driver_fullname) %>%
  summarize(total_wins = sum(if_else(position == 1, TRUE, FALSE), na.rm = TRUE), # calculate total driver wins
            total_podium = sum(if_else(position %in% c(1:3), TRUE, FALSE), na.rm = TRUE), # calculate total driver podiums
            last_race = max(year), # drivers last year raced
            total_races = n(), # total races driver has entered
            win_perc = paste0(formatC((total_wins / total_races)*100, format="f", digits=1), "%"), # format as percentage
            podium_perc = paste0(formatC((total_podium / total_races)*100, format="f", digits=1), "%")) %>%
  ungroup() %>%
  top_n(10, total_wins) %>% # limit to top 10 by total wins number
  arrange(desc(total_wins)) %>% # arange by total wins
  print.data.frame() # print data frame


### Pole to Win Conversion / None Pole to Win Rate ###
# make new columns for pole conversions and none pole wins
master_results$pole_conversion <- ifelse(master_results$grid==1 & master_results$positionOrder==1, 1, 0)
master_results$none_pole_win <- ifelse(master_results$grid!=1 & master_results$positionOrder==1, 1, 0)

# calculate total wins in both cases, and percentage of total races for both, and display
master_results %>%
  group_by(driver_fullname) %>%
  summarize(total_races = n(),
            pole_conversions = sum(if_else(pole_conversion == 1, TRUE, FALSE), na.rm = TRUE),
            none_pole_wins = sum(if_else(none_pole_win == 1, TRUE, FALSE), na.rm = TRUE),
            pole_conv_perc = paste0(formatC((pole_conversions / total_races)*100, format="f", digits=1), "%"),
            none_pole_perc = paste0(formatC((none_pole_wins / total_races)*100, format="f", digits=1), "%"),
            total_win_perc = paste0(formatC(((pole_conversions+none_pole_wins) / total_races)*100, format="f", digits=1), "%")) %>%
  ungroup() %>%
  top_n(10, pole_conversions) %>%
  arrange(desc(total_races)) %>%
  print.data.frame()

### calculate total championships for each driver, plot H bar chart ###
total_champs <- master_results %>%
  group_by(year, driver_fullname) %>%
  summarise(total_points = sum(points, na.rm = TRUE)) %>% 
  top_n(1, total_points) %>%
  ggplot(aes(fct_rev(fct_infreq(driver_fullname))))+
  geom_bar(position = "dodge", fill = "#FF1801")+
  labs(title = "Total Championships",
       x = "Drivers",
       y = "No. of Championships")+coord_flip()

ggsave("./EDA_total_championships_by_driver.png", plot=total_champs)


### filter for wins only, calculate average age of drivers by year, plot line plot ###
avg_winner_age <- master_results %>% 
  filter(position==1) %>% 
  group_by(year) %>% 
  summarise(avg_age=mean(driver_age)) %>% 
  ggplot(aes(year, avg_age))+
  geom_line( color="black", size=0.5, alpha=0.9, linetype=1)+
  ylim(15, 40)+
  geom_point(color="#FF1801")+
  labs(title = "Average Age of Race Winners over Time",
     x = "Year",
     y = "Average age of race winners")  

ggsave("./EDA_average_winners_age.png", plot=avg_winner_age)


### Filter for 3 constructors, plot Bar plot of positions over past 5 seasons ###
avg_constructor_positions <- master_results %>% 
  filter(constructor_name %in% c("Ferrari", "Red Bull", "Mercedes"), year>2017) %>%
  select(year, constructor_name, positionOrder) %>% 
  ggplot(aes(x=factor(year), y=positionOrder, fill=constructor_name, dodge=constructor_name))+
  stat_boxplot(geom ='errorbar')+
  geom_boxplot()+
  scale_fill_manual(values=c("#DC0000","#00D2BE","#0600EF"))+
  ylim(1, 25)+
  labs(title = "Race Finishing Positions for Ferrari, Mercedes & Red Bull over last 5 seasons",
       x = "Season",
       y = "Finishing Race Position")
  
ggsave("./EDA_avg_constructor_wins.png", plot=avg_constructor_positions)


### Map of Globe Frequency of number of races ###

# Set euro countries to variable
european_countries <- c("Albania", "Andorra", "Armenia", "Austria", "Azerbaijan",
                        "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
                        "Croatia", "Cyprus", "Czechia","Denmark","Estonia", 
                        "France","Georgia", "Germany", "Greece","Hungary", 
                        "Ireland", "Italy", "Kosovo", "Latvia","Liechtenstein", 
                        "Lithuania", "Luxembourg","Malta","Moldova","Monaco","Montenegro",
                        "Macedonia", "Netherlands","Poland","Portugal","Romania",
                        "San Marino","Serbia","Slovakia","Slovenia","Spain",
                        "Switzerland","Turkey","Ukraine","UK","Vatican")

# get world map data from ggplot, filter for above euro countries
world_map <- map_data("world") %>%
  filter(region %in% european_countries)


# calculate Hamilton wins and group by location
hamilton_circuit_wins <- master_results %>% 
  filter(driverRef=="hamilton", position == 1) %>% 
  group_by(circuitRef, lat, lng, country) %>%
  summarize(win_count = n_distinct(raceId)) %>% 
  ungroup() %>% 
  select(circuitRef, win_count)

# calculate Hamilton races and  filter for euro countries, group by location, join on wins
hamilton_circuit_races <- master_results %>% 
  filter(driverRef=="hamilton", country %in% european_countries) %>% 
  group_by(circuitRef, lat, lng, country) %>%
  summarize(race_count = n_distinct(raceId)) %>% 
  ungroup() %>% 
  left_join(hamilton_circuit_wins, by = "circuitRef")

# remove wins dataframe
rm(hamilton_circuit_wins)

# Replace NA with 0 in hamilton_circuit_races
hamilton_circuit_races[is.na(hamilton_circuit_races)] <- 0

# Calculate Hamilton Win Ratio
hamilton_circuit_races$win_ratio <- round(hamilton_circuit_races$win_count / hamilton_circuit_races$race_count, 2)

# Plot map with sized points for hamilton win ratio per track in Europe
ggplot() +
  geom_map(data = world_map, map = world_map,
           aes(x = long, y = lat, map_id = region),
           fill = "black", color = "#ffffff", size = 0.7, alpha=0.25)+
  geom_point(data = hamilton_circuit_races, aes(x = lng, y = lat, size = win_ratio), 
             col = "red")+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())+
  labs(title = "Lewis Hamilton Win Ratio in European Races",
       x = "",
       y = "")


### filter for UK Wins, group by driver nationality and plot horizontal bar chart ###
british_gp_winners <- master_results %>%
  filter(country == "UK", position == 1) %>% #, statusId==1) %>% 
  select(year, driver_nationality) %>% 
  group_by(year, driver_nationality) %>%
  summarise(count_nt = n()) %>% 
  ggplot(aes(fct_rev(fct_infreq(driver_nationality)))) +
  geom_bar(position = "dodge", fill = "#00D2BE")+
  labs(title = "Total Wins at Silverstone (UK) by Driver Nationality",
       x = "Driver Nationality",
       y = "Number of Wins")+coord_flip()

ggsave("./EDA_silverstone_winners.png", plot=british_gp_winners)

