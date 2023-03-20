"""Evaluation for test_building_heating_demand_dummy_heater."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from hisim import log
from ordered_set import OrderedSet


with open("C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\Nineth_try_only_DE_with_15min_timesteps_Aachen_weather\\test_building_heating_demand_dummy_heater_DE_energy_needs1.csv", "r") as myfile:
    lines = myfile.readlines()[1:]
    print(len(lines))

if len(lines)  == 0:
    pass
    print("file is empty. nothing to analyze.")
else:
    building_codes = []
    types= []
    types_details = []
    heat_demands_hisim = []
    heat_demands_tabula = []
    ratios = []

    for index,line in enumerate(lines):
        splitted_line = line.split(";")
        building_code = splitted_line[0]
        type = ".".join(building_code.split(".")[0:-2])
        type_detail = ".".join(building_code.split(".")[-2:])
        heat_demand_hisim = float(splitted_line[1])
        log.information(str(building_code))
        log.information(str(heat_demand_hisim))
        heat_demand_tabula = float(splitted_line[2])
        ratio_hp_tabula = splitted_line[3]
        log.information(str(ratio_hp_tabula))
        ratio_hp_tabula_floats = float(ratio_hp_tabula)
        building_codes.append(building_code)
        types.append(type)
        types_details.append(type_detail)
        heat_demands_hisim.append(heat_demand_hisim)
        heat_demands_tabula.append(heat_demand_tabula)
        ratios.append(ratio_hp_tabula_floats)


    tabula_types = list(OrderedSet(types))
    log.information(str(tabula_types))
    list_of_indices = []
    list_of_ratios = []
    list_of_heat_demand_hisim = []
    list_of_heat_demand_tabula = []
    list_of_type_details = []
    for index_1, tabula_type in enumerate(tabula_types):
        list_of_indices_of_one_type = []
        list_of_ratios_of_one_type = []
        list_of_heat_demand_hisim_of_one_type = []
        list_of_heat_demand_tabula_of_one_type = []
        list_of_type_details_of_one_type = []

        for index_2, tabula_type2 in enumerate(types):
            if tabula_type == tabula_type2:
                list_of_indices_of_one_type.append(index_2)
                list_of_ratios_of_one_type.append(ratios[index_2])
                list_of_heat_demand_hisim_of_one_type.append(heat_demands_hisim[index_2])
                list_of_heat_demand_tabula_of_one_type.append(heat_demands_tabula[index_2])
                list_of_type_details_of_one_type.append(types_details[index_2])
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
        plt.savefig(f"C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\Nineth_try_only_DE_with_15min_timesteps_Aachen_weather\\Evaluation\\{tabula_type}.png")
        plt.close()

        list_of_indices.append(list_of_indices_of_one_type)
        list_of_ratios.append(list_of_ratios_of_one_type)


    max_length_of_list_of_ratios = max(len(list) for list in list_of_ratios)

    for index, list_in_list_of_ratios in enumerate(list_of_ratios):
        if len(list_in_list_of_ratios) < max_length_of_list_of_ratios:
            length_difference = max_length_of_list_of_ratios - len(list_in_list_of_ratios)
            list_of_ratios[index] = list_in_list_of_ratios +  (list(np.repeat(np.nan, length_difference)))


    dictionary_countries_and_indices = dict(zip(tabula_types, list_of_indices))
    dictionary_countries_and_ratios = dict(zip(tabula_types, list_of_ratios))

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
    plt.savefig(f"C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\Nineth_try_only_DE_with_15min_timesteps_Aachen_weather\\Evaluation\\All_DE_types_ratio.png")




