import pandas as pd
import glob

allFiles = glob.glob("data/real/*.csv")
df = pd.concat((pd.read_csv(f) for f in allFiles))
df.to_csv("data/real.csv", sep=',')