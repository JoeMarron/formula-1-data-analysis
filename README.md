
# Formula 1 Machine Learning in R

***Work in Progress*** \
Implementing a machine learning model on Formula 1 data to predict race winners. First time using R, as part of the 'R for Data Science' module in the MSc Data Science & Financial Technology from University of London.

## Data Preprocessing
The following CSV files sourced from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) have been cleansed and combined into a final dataset, some examples of the columns shown below. Some columns have also been calculated, such as driver age.

**Files** \
*Fact Tables*
- results.csv
*Dimension Tables*
- circuits.csv
- constructors.csv
- drivers.csv
- races.csv
- status.csv

#### Example Output

| year | round | grid | positionOrder | points | laps | milliseconds | fastestLap | fastestLapSpeed | driverRef       | driver_nationality | driver_age | constructor_name | circuitRef |
| ---- | ----- | ---- | ------------- | ------ | ---- | ------------ | ---------- | --------------- | --------------- | ------------------ | ---------- | ---------------- | ---------- |
| 2022 | 1     | 1    | 1             | 26     | 57   | 5853584      | 51         | 206.018         | leclerc         | Monegasque         | 24         | Ferrari          | bahrain    |
| 2022 | 1     | 3    | 2             | 18     | 57   | 5859182      | 52         | 203.501         | sainz           | Spanish            | 27         | Ferrari          | bahrain    |
| 2022 | 1     | 5    | 3             | 15     | 57   | 5863259      | 53         | 202.469         | hamilton        | British            | 37         | Mercedes         | bahrain    |
| 2022 | 1     | 9    | 4             | 12     | 57   | 5864795      | 56         | 202.313         | russell         | British            | 24         | Mercedes         | bahrain    |
| 2022 | 1     | 7    | 5             | 10     | 57   | 5868338      | 53         | 201.641         | kevin_magnussen | Danish             | 29         | Haas F1 Team     | bahrain    |
| 2022 | 1     | 6    | 6             | 8      | 57   | 5869703      | 53         | 201.691         | bottas          | Finnish            | 32         | Alfa Romeo       | bahrain    |
| 2022 | 1     | 11   | 7             | 6      | 57   | 5873007      | 53         | 200.63          | ocon            | French             | 25         | Alpine F1 Team   | bahrain    |
| 2022 | 1     | 16   | 8             | 4      | 57   | 5873970      | 53         | 200.642         | tsunoda         | Japanese           | 21         | AlphaTauri       | bahrain    |
| 2022 | 1     | 8    | 9             | 2      | 57   | 5875974      | 44         | 201.412         | alonso          | Spanish            | 40         | Alpine F1 Team   | bahrain    |
| 2022 | 1     | 15   | 10            | 1      | 57   | 5876648      | 39         | 201.512         | zhou            | Chinese            | 22         | Alfa Romeo       | bahrain    |

## Exploratory Data Analysis
I explored the data in detail with a couple examples of results shown below. All code can be found in the data_cleansing_and_EDA.R file.

#### Home Advantage in F1
Below shows the code and subsequent graph showing the Silverstone GP winners grouped by nationality. This demonstrates a potential for home race advantage, although some of this may be bias as Hamilton makes up a substantial amount of the 'British' wins (8/29) and has won a large proportion of all the races in the last ~15 years.

```
british_gp_winners <- master_results %>%
  filter(country == "UK", position == 1) %>%
  select(year, driver_nationality) %>% 
  group_by(year, driver_nationality) %>%
  summarise(count_nt = n()) %>% 
  ggplot(aes(fct_rev(fct_infreq(driver_nationality)), fill=as.factor(ifelse(driver_nationality=="British","Highlight","Normal"))))+
  geom_bar(position = "dodge", show.legend=FALSE)+
  scale_fill_manual(values=c("#C8102E","#8898AC"))+
  labs(title = "Total Wins at Silverstone (UK) by Driver Nationality",
       x = "Driver Nationality",
       y = "Number of Wins")+coord_flip()
```

![silverstone_winners](https://github.com/joemarron/formula-1-machine-learning/blob/main/EDA/EDA_silverstone_winners.png)

#### Constructor Race Wins by Season
Below shows how three competitive constructors (Ferrari, Mercedes and Red Bull) average finishing positions change across seasons from 2018-2022. This demonstrates how constructor information will be crucial for a machine learning model in predicting race winners.

```
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
```

In 2021, the lower quartile between Mercedes and Red Bull is close, with more overall variance in Red Bulls finishing position. This is backed up as 2021 was the most competitive season since the dawn of the hybrid era (2014), with Mercedes winning the constrcutors title, due to more consistantly high finishing positions, but Verstappen ultimately *won* the drivers championship, likely explaining the slightly lower lower quartile for Red Bull.

![cons_positions](https://github.com/joemarron/formula-1-machine-learning/blob/main/EDA/EDA_avg_constructor_wins.png)

## TBC
Next, I will look at how to apply machine learning methods to this F1 data to try and predict race winners.

