"""Evaluation for test_building_heating_demand_dummy_heater."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from hisim import log
from ordered_set import OrderedSet


        
# directory = "C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\\Twelwth_try_only_DE_Aachen_fixed_window_areas"
# directory = "C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\Eleventh_try_only_DE_all_time_resolutions_Aachen_fixed_idealized_heater"
directory = "C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\Thirteenth_try_DE_Aachen_heater_occupancy_window_areas_fixed"
read_csv_file = pd.read_csv(directory + "\\test_building_heating_demand_dummy_heater_DE_energy_needs1.csv", sep=";", header=None)
read_csv_file.to_excel(directory + "\\test_building_heating_demand_dummy_heater_DE_energy_needs11.xlsx", header=True)

xl = pd.read_excel(directory + "\\test_building_heating_demand_dummy_heater_DE_energy_needs11.xlsx")


building_codes = []
types= []
types_details = []
heat_demands_hisim = []
heat_demands_tabula = []
ratios = []
areas = []

xl_outliers = pd.DataFrame(data=None, columns=xl.columns)
for index, row in xl.iterrows():
    building_code = row[0]
    tabula_type = ".".join(building_code.split(".")[0:-2])
    type_detail = ".".join(building_code.split(".")[-2:])
    hisim_heat_demand = row[2]
    tabula_heat_demand = row[3]
    ratio_heat_demand_hisim_tabula = hisim_heat_demand/tabula_heat_demand
    area = row[30]

    building_codes.append(building_code)
    types.append(tabula_type)
    types_details.append(type_detail)
    heat_demands_hisim.append(hisim_heat_demand)
    heat_demands_tabula.append(tabula_heat_demand)
    ratios.append(ratio_heat_demand_hisim_tabula)
    areas.append(area)
    if ratio_heat_demand_hisim_tabula <=0.70 or ratio_heat_demand_hisim_tabula >= 1.30:
        xl_outliers.loc[len(xl_outliers)] = row
        

min_ratio =min(ratios)
max_ratio = max(ratios)
print("min ratio " ,min_ratio )
print("max ratio ",max_ratio)
print(xl_outliers)

xl_outliers.to_excel(directory + "\\buildings_with_ratio_outliers.xlsx")

# nach types sortieren
tabula_types = list(OrderedSet(types))
list_of_indices = []
list_of_ratios = []
list_of_heat_demand_hisim = []
list_of_heat_demand_tabula = []
list_of_type_details = []
list_of_areas = []
for index_1, tabula_type in enumerate(tabula_types):
    list_of_indices_of_one_type = []
    list_of_ratios_of_one_type = []
    list_of_heat_demand_hisim_of_one_type = []
    list_of_heat_demand_tabula_of_one_type = []
    list_of_type_details_of_one_type = []
    list_of_areas_of_one_type = []

    for index_2, tabula_type2 in enumerate(types):
        if tabula_type == tabula_type2:
            list_of_indices_of_one_type.append(index_2)
            list_of_ratios_of_one_type.append(ratios[index_2])
            list_of_heat_demand_hisim_of_one_type.append(heat_demands_hisim[index_2])
            list_of_heat_demand_tabula_of_one_type.append(heat_demands_tabula[index_2])
            list_of_type_details_of_one_type.append(types_details[index_2])
            list_of_areas_of_one_type.append(areas[index_2])
    minimum = min(min(list_of_heat_demand_hisim_of_one_type), min(list_of_heat_demand_tabula_of_one_type))
    maximum = max(max(list_of_heat_demand_hisim_of_one_type), max(list_of_heat_demand_tabula_of_one_type))

    plt.figure(figsize=(15,9))
    plt.scatter(x=list_of_heat_demand_hisim_of_one_type, y=list_of_heat_demand_tabula_of_one_type, c="red")
    plt.plot([minimum, maximum],[minimum,maximum], "-")

    index = 0
    for demand_hisim, demand_tabula in zip(list_of_heat_demand_hisim_of_one_type, list_of_heat_demand_tabula_of_one_type):

        label = f"{list_of_type_details_of_one_type[index]} \n Ratio: {list_of_ratios_of_one_type[index]}"

        plt.annotate(label, # this is the text
                    (demand_hisim, demand_tabula),# these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center',
                    fontsize=8) # horizontal alignment can be left, right or center
        index = index + 1
    plt.title(f"Validation of Heating Demand of for Tabula Type {tabula_type}", fontsize=14)
    plt.xlabel("Heating Demand given by Idealized Electric Heater HiSim [kWh/a*m2]", fontsize=12)
    plt.ylabel("Heating Demand given by Tabula [kWh/a*m2]", fontsize=12)
    plt.xticks(fontsize=10)
    plt.savefig(directory + f"\\Evaluation\\{tabula_type}.png")
    plt.close()

    list_of_indices.append(list_of_indices_of_one_type)
    list_of_ratios.append(list_of_ratios_of_one_type)
    list_of_areas.append(list_of_areas_of_one_type)



max_length_of_list_of_ratios = max(len(list) for list in list_of_ratios)

for index, list_in_list_of_ratios in enumerate(list_of_ratios):
    if len(list_in_list_of_ratios) < max_length_of_list_of_ratios:
        length_difference = max_length_of_list_of_ratios - len(list_in_list_of_ratios)
        list_of_ratios[index] = list_in_list_of_ratios +  (list(np.repeat(np.nan, length_difference)))

max_length_of_list_of_areas = max(len(list) for list in list_of_areas)

for index, list_in_list_of_areas in enumerate(list_of_areas):
    if len(list_in_list_of_areas) < max_length_of_list_of_areas:
        length_difference = max_length_of_list_of_areas - len(list_in_list_of_areas)
        list_of_areas[index] = list_in_list_of_areas +  (list(np.repeat(np.nan, length_difference)))



dictionary_countries_and_indices = dict(zip(tabula_types, list_of_indices))
dictionary_countries_and_ratios = dict(zip(tabula_types, list_of_ratios))
dictionary_countries_and_areas = dict(zip(tabula_types, list_of_areas))

df = pd.DataFrame(dictionary_countries_and_ratios)

df.index.name="Index Name"
df.columns.name="Tabula Types"
fig =plt.figure(figsize=(16,10))

seaborn.boxplot(data=df)
seaborn.swarmplot(data=df, color="grey", size=4)
fig.autofmt_xdate(rotation=45)
plt.xticks(fontsize=9)
plt.axhline(y=1, color="red")
plt.title("Validation of Heating Demand of German Houses", fontsize=14)
plt.ylabel("Ratio of Heating Demand given by HiSim/Tabula", fontsize=12)
plt.xlabel("Tabula Building Codes", fontsize=12)
plt.tight_layout()
plt.savefig(directory + "\\Evaluation\\All_DE_types_ratio.png")
plt.close()

df = pd.DataFrame(dictionary_countries_and_areas)

df.index.name="Index Name"
df.columns.name="Tabula Types"
fig =plt.figure(figsize=(16,10))

seaborn.boxplot(data=df)
seaborn.swarmplot(data=df, color="grey", size=4)
fig.autofmt_xdate(rotation=45)
plt.xticks(fontsize=9)
plt.axhline(y=1, color="red")
plt.title("Validation of Heating Demand of German Houses", fontsize=14)
plt.ylabel("Areas of Buildings [m2]", fontsize=12)
plt.xlabel("Tabula Building Codes", fontsize=12)
plt.tight_layout()
plt.savefig(directory + "\\Evaluation\\All_DE_types_area.png")
plt.close()

fig = plt.figure(figsize=(8,8))
seaborn.boxplot(data=xl[4])
seaborn.swarmplot(data=xl[4], color="grey", size=4)
plt.axhline(y=1, color="red")
plt.xlabel("Tabula DE buildings", fontsize=12)
plt.ylabel("Ratio of Heating Demand given by HiSim/Tabula", fontsize=12)
textstr = '\n'.join((
    f'Min Ratio {np.round(min_ratio,2)}',
    f'Max Ratio {np.round(max_ratio,2)}'))
plt.text(0.2, 0.95, textstr, bbox= dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.savefig(directory + "\\Evaluation\\Swarmplot_All_DE_Ratios_Together.png")
plt.show()


