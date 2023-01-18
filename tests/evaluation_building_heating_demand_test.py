import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from collections import OrderedDict
# with open("C:\\Users\\k.rieck\\Desktop\\test_building_heating_demand_all_tabula_energy_needs_with_tabula_areas_without_Area_Zero.txt", "r") as myfile:
with open("C:\\Users\\k.rieck\\Desktop\\test_building_heating_demand_all_tabula_energy_needs.txt", "r") as myfile:
    lines = myfile.readlines()[1:]
    print(len(lines))
with open("C:\\Users\\k.rieck\\Desktop\\test_building_heating_demand_all_tabula_energy_needs_with_tabula_areas_without_Area_Zero.txt", "r") as myfile:
    lines1 = myfile.readlines()[1:]
    print(len(lines1))

building_codes = []
ratios = []
countries = []

for index,line in enumerate(lines):
    splitted_line = line.split(";")
    building_code = splitted_line[0]
    country = building_code[0:2]

    ratio_hp_tabula = splitted_line[-1]
    ratio_hp_tabula_floats = float(ratio_hp_tabula)
    building_codes.append(building_code)
    ratios.append(ratio_hp_tabula_floats)
    countries.append(country)

building_codes1 = []
ratios1 = []
countries1 = []

for index,line in enumerate(lines1):
    splitted_line = line.split(";")
    building_code = splitted_line[0]
    country = building_code[0:2]

    ratio_hp_tabula = splitted_line[-1]
    ratio_hp_tabula_floats = float(ratio_hp_tabula)
    building_codes1.append(building_code)
    ratios1.append(ratio_hp_tabula_floats)
    countries1.append(country)

tabula_countries = list(set(countries))
list_of_indices = []
list_of_ratios = []
for index_1, tabula_country in enumerate(tabula_countries):
    list_of_indices_of_one_country = []
    list_of_ratios_of_one_country = []
    for index_2, country in enumerate(countries):
        if tabula_country == country:
            list_of_indices_of_one_country.append(index_2)
            list_of_ratios_of_one_country.append(ratios[index_2])

    list_of_indices.append(list_of_indices_of_one_country)
    list_of_ratios.append(list_of_ratios_of_one_country)

tabula_countries1 = list(set(countries1))
list_of_indices1 = []
list_of_ratios1 = []
for index_1, tabula_country in enumerate(tabula_countries1):
    list_of_indices_of_one_country = []
    list_of_ratios_of_one_country = []
    for index_2, country in enumerate(countries1):
        if tabula_country == country:
            list_of_indices_of_one_country.append(index_2)
            list_of_ratios_of_one_country.append(ratios1[index_2])

    list_of_indices1.append(list_of_indices_of_one_country)
    list_of_ratios1.append(list_of_ratios_of_one_country)


max_length_of_list_of_ratios = max(len(list) for list in list_of_ratios)
max_length_of_list_of_ratios1 = max(len(list) for list in list_of_ratios1)

for index, list_in_list_of_ratios in enumerate(list_of_ratios):
    if len(list_in_list_of_ratios) < max_length_of_list_of_ratios:
        length_difference = max_length_of_list_of_ratios - len(list_in_list_of_ratios)
        list_of_ratios[index] = list_in_list_of_ratios +  (list(np.repeat(np.nan, length_difference)))

for index, list_in_list_of_ratios1 in enumerate(list_of_ratios1):
    if len(list_in_list_of_ratios1) < max_length_of_list_of_ratios1:
        length_difference = max_length_of_list_of_ratios1 - len(list_in_list_of_ratios1)
        list_of_ratios1[index] = list_in_list_of_ratios1 +  (list(np.repeat(np.nan, length_difference)))

dictionary_countries_and_indices = dict(zip(tabula_countries, list_of_indices))
dictionary_countries_and_ratios = dict(zip(tabula_countries, list_of_ratios))
dictionary_countries_and_indices1 = dict(zip(tabula_countries1, list_of_indices1))
dictionary_countries_and_ratios1 = dict(zip(tabula_countries1, list_of_ratios1))

df = pd.DataFrame(dictionary_countries_and_ratios, index=["first_try"]*584)
df1 = pd.DataFrame(dictionary_countries_and_ratios1, index=["second_try"]*465)

df2 =pd.concat([df, df1])
df2.index.name="Index Name"
df2.columns.name="Countries"
plt.figure(figsize=(5,5))
# df.boxplot(column=tabula_countries)
# df2.boxplot(column=tabula_countries1, by="Index Name")
# print(df2)
# seaborn.boxplot(data=df2)
# seaborn.swarmplot(data=df2)
plt.axhline(y=1, color="red")
plt.title("Heating Test First Try")
plt.ylabel("Ratio energy need HP/Tabula")
plt.axis([-1,22,0,3])
plt.xlabel("Country")
plt.show()