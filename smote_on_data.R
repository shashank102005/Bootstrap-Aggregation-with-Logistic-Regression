DigitalBreathTestData2014 = read.csv("f:/University of Cincinnati/Fall 2015 Courses/Data Management/Open Data/DigitalBreathTestData2014.txt")
DigitalBreathTestData2014$Year = NULL

New = vector("numeric")

for( i in 1:nrow(DigitalBreathTestData2014))
{
  if(DigitalBreathTestData2014$BreathAlcoholLevel[i] > 35)
    New[i] = 1
  else
    New[i] = 0
}

df_new = data.frame(New)

df_dtgiBreath = cbind(DigitalBreathTestData2014,df_new)
colnames(df_dtgiBreath)[8]='Target'

samp = sample(nrow(df_dtgiBreath),nrow(df_dtgiBreath)*0.20)
df_test = df_dtgiBreath[samp,] 
df_train = df_dtgiBreath[-samp,]

df_train$Target = as.factor(df_train$Target)

df_train_bal = SMOTE(Target ~ . - BreathAlcoholLevel, df_train, perc.over = 600,perc.under=100)

str(df_train)

df_train_bal = df_train_bal[sample(nrow(df_train_bal),nrow(df_train_bal)),]

