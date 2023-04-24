"""Evaluation for test_building_heating_demand_dummy_heater."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from hisim import log
from ordered_set import OrderedSet
from typing import Any
import os


directory = "C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\Fifteenth_try_only_DE_solar_gains_and_internal_gains_from_tabula\\"
#directory = "C:\\Users\\k.rieck\\Documents\\Software_and_Tools_Documentation\\HiSim\\Tests\\Heating_Test\\Twelwth_try_only_DE_Aachen_fixed_window_areas\\"
#file = "test_building_heating_demand_dummy_heater_DE_energy_needs1.xlsx"
file = "test_building_heating_demand_dummy_heater_DE_energy_needs0.xlsx"


class TestBuildingHeatingDemandEvaluation:
    """TestBuildingHeatingDemandEvalation."""

    def __init__(self, directory_path: str, file_name: str) -> None:
        self.directory_path = directory_path
        self.data_file_path = self.directory_path + file_name
        self.evaluation_of = "Heat Demand"
        print(f"{self.evaluation_of} of HiSim vs Tabula will be evaluated.")
        if (
            os.path.exists(self.directory_path + f"\\Evaluation_{self.evaluation_of}")
            is False
        ):
            os.makedirs(self.directory_path + f"\\Evaluation_{self.evaluation_of}")

        xl = self.read_data()
        (
            building_codes,
            tabula_types,
            tabula_type_details,
            areas,
            hisim_values,
            tabula_values,
            ratios_hisim_vs_tabula,
        ) = self.get_data_from_dataframe(dataframe=xl)

        dataframe_outliers = self.get_outliers_and_min_and_max_from_list(
            dataframe=xl, list_to_be_checked=ratios_hisim_vs_tabula
        )
        (
            list_of_indices,
            list_of_ratios,
            sorted_tabula_types,
        ) = self.make_graph_for_one_type(
            tabula_types=tabula_types,
            ratios=ratios_hisim_vs_tabula,
            values_hisim=hisim_values,
            values_tabula=tabula_values,
            types_details=tabula_type_details,
            areas=areas,
            evaluation=self.evaluation_of,
        )
        list_of_ratios_same_length = self.give_lists_the_same_length_for_plotting(
            list_of_lists=list_of_ratios
        )

        df_tabula_types_and_ratios = self.create_dictionary_and_dataframe(
            list_one=sorted_tabula_types,
            list_two=list_of_ratios_same_length,
            column_name="Tabula Types",
        )
        self.make_plot_for_all_tabula_types(
            dataframe=df_tabula_types_and_ratios, evaluation=self.evaluation_of
        )
        self.make_swarm_boxplot(
            data=ratios_hisim_vs_tabula, evaluation=self.evaluation_of
        )

    def read_data(self) -> pd.DataFrame:

        data_file_ending = self.data_file_path.split(".")[-1]
        rest_of_data_file = ".".join(self.data_file_path.split(".")[0:-1])

        if data_file_ending == "csv":
            read_file = pd.read_csv(
                filepath_or_buffer=self.data_file_path, sep=";", header=None
            )
            excel_file = rest_of_data_file + ".xlsx"
            read_file.to_excel(excel_file, header=True)
        elif data_file_ending == "xlsx":
            excel_file = self.data_file_path
        else:
            raise TypeError("File format should be csv or xlsx.")

        xl = pd.read_excel(excel_file)
        return xl

    def get_data_from_dataframe(self, dataframe: pd.DataFrame) -> list:

        building_codes = dataframe["Building Code"]
        tabula_types = []
        tabula_type_details = []
        for code in building_codes:

            tabula_type = ".".join(code.split(".")[0:-2])
            tabula_type_detail = ".".join(code.split(".")[-2:])
            tabula_types.append(tabula_type)
            tabula_type_details.append(tabula_type_detail)

        areas = dataframe["Conditioned Floor Area [m2]"]

        (
            hisim_values,
            tabula_values,
            ratios_hisim_vs_tabula,
        ) = self.decide_what_data_to_evaluate(
            evaluation=self.evaluation_of, dataframe=dataframe
        )

        return (
            building_codes,
            tabula_types,
            tabula_type_details,
            areas,
            hisim_values,
            tabula_values,
            ratios_hisim_vs_tabula,
        )

    def decide_what_data_to_evaluate(self, evaluation: str, dataframe: pd.DataFrame):

        if evaluation == "Heat Demand":
            hisim_values = dataframe[
                "Heating demand for heating from Dummy Heater [kWh/(a*m2)]"
            ]
            tabula_values = dataframe[
                "Heating demand for heating from TABULA [kWh/(a*m2)]"
            ]
            ratios_hisim_vs_tabula = dataframe["Ratio HiSim/TABULA heating demand"]

        elif evaluation == "Internal Gains":
            hisim_values = dataframe["Internal gains from Occupancy [kWh/(a*m2)]"]
            tabula_values = dataframe["Internal gains from TABULA [kWh/(a*m2)]"]
            ratios_hisim_vs_tabula = dataframe["Ratio HiSim/TABULA internal gains"]

        elif evaluation == "Solar Gains":
            hisim_values = dataframe["Solar gains from Windows [kWh/(a*m2)]"]
            tabula_values = dataframe["Solar gains from TABULA [kWh/(a*m2)]"]
            ratios_hisim_vs_tabula = dataframe["Ratio HiSim/TABULA solar gains"]
        else:
            raise KeyError("this evaluation key is not found.")

        return hisim_values, tabula_values, ratios_hisim_vs_tabula

    def get_outliers_and_min_and_max_from_list(
        self, dataframe: pd.DataFrame, list_to_be_checked: list
    ):

        dataframe_outliers = pd.DataFrame(data=None, columns=dataframe.columns)

        for index, row in dataframe.iterrows():

            if list_to_be_checked[index] <= 0.80 or list_to_be_checked[index] >= 1.20:
                dataframe_outliers.loc[len(dataframe_outliers)] = row
        
        self.min_ratio = min(list_to_be_checked)
        self.max_ratio = max(list_to_be_checked)
        self.median = np.median(list_to_be_checked)
        self.mean = np.mean(list_to_be_checked)
        self.std = np.std(list_to_be_checked)
        print(list_to_be_checked)
        print(self.mean)
        dataframe_outliers.to_excel(
            self.directory_path
            + f"\\Evaluation_{self.evaluation_of}\\buildings_with_ratio_outliers.xlsx"
        )
        return list(OrderedSet(dataframe_outliers))

    def make_graph_for_one_type(
        self,
        tabula_types: list,
        ratios: list,
        values_hisim: list,
        values_tabula: list,
        types_details: list,
        areas: list,
        evaluation: str,
    ):

        sorted_tabula_types = list(OrderedSet(tabula_types))
        list_of_indices = []
        list_of_ratios = []
        for index_1, tabula_type in enumerate(sorted_tabula_types):
            list_of_indices_of_one_type = []
            list_of_ratios_of_one_type = []
            list_of_values_hisim_of_one_type = []
            list_of_values_tabula_of_one_type = []
            list_of_type_details_of_one_type = []

            for index_2, tabula_type2 in enumerate(tabula_types):
                if tabula_type == tabula_type2:
                    list_of_indices_of_one_type.append(index_2)
                    list_of_ratios_of_one_type.append(ratios[index_2])
                    list_of_values_hisim_of_one_type.append(values_hisim[index_2])
                    list_of_values_tabula_of_one_type.append(values_tabula[index_2])
                    list_of_type_details_of_one_type.append(types_details[index_2])

            minimum = min(
                min(list_of_values_hisim_of_one_type),
                min(list_of_values_tabula_of_one_type),
            )
            maximum = max(
                max(list_of_values_hisim_of_one_type),
                max(list_of_values_tabula_of_one_type),
            )

            plt.figure(figsize=(15, 9))
            plt.scatter(
                x=list_of_values_hisim_of_one_type,
                y=list_of_values_tabula_of_one_type,
                c="red",
            )
            plt.plot([minimum, maximum], [minimum, maximum], "-")

            index = 0
            for demand_hisim, demand_tabula in zip(
                list_of_values_hisim_of_one_type, list_of_values_tabula_of_one_type
            ):

                label = f"{list_of_type_details_of_one_type[index]} \n Ratio: {list_of_ratios_of_one_type[index]}"

                plt.annotate(
                    label,  # this is the text
                    (
                        demand_hisim,
                        demand_tabula,
                    ),  # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 10),  # distance from text to points (x,y)
                    ha="center",
                    fontsize=8,
                )  # horizontal alignment can be left, right or center
                index = index + 1
            plt.title(
                f"Validation of {evaluation} for Tabula Type {tabula_type}", fontsize=14
            )
            plt.xlabel(
                f"{evaluation} given by Idealized Electric Heater HiSim [kWh/a*m2]",
                fontsize=12,
            )
            plt.ylabel(f"{evaluation} given by Tabula [kWh/a*m2]", fontsize=12)
            plt.xticks(fontsize=10)
            plt.savefig(
                self.directory_path
                + f"\\Evaluation_{self.evaluation_of}\\{tabula_type}.png"
            )
            plt.close()

            list_of_indices.append(list_of_indices_of_one_type)
            list_of_ratios.append(list_of_ratios_of_one_type)

        return list_of_indices, list_of_ratios, sorted_tabula_types

    def give_lists_the_same_length_for_plotting(self, list_of_lists: list[list]):

        max_length_of_list_of_ratios = max(len(list) for list in list_of_lists)

        for index, list_in_list_of_ratios in enumerate(list_of_lists):
            if len(list_in_list_of_ratios) < max_length_of_list_of_ratios:
                length_difference = max_length_of_list_of_ratios - len(
                    list_in_list_of_ratios
                )
                list_of_lists[index] = list_in_list_of_ratios + (
                    list(np.repeat(np.nan, length_difference))
                )
        return list_of_lists

    def create_dictionary_and_dataframe(
        self, list_one: list, list_two: list, column_name: str
    ):

        dictionary = dict(zip(list_one, list_two))
        df = pd.DataFrame(dictionary)
        df.index.name = "Index Name"
        df.columns.name = column_name
        return df

    def make_plot_for_all_tabula_types(self, dataframe: pd.DataFrame, evaluation: str):

        fig = plt.figure(figsize=(16, 10))
        print(dataframe)
        seaborn.boxplot(data=dataframe)
        seaborn.swarmplot(data=dataframe, color="grey", size=4)
        fig.autofmt_xdate(rotation=45)
        plt.xticks(fontsize=9)
        plt.axhline(y=1, color="red")
        plt.title(f"Validation of {evaluation} of German Houses", fontsize=14)
        plt.ylabel(f"Ratio of {evaluation} given by HiSim/Tabula", fontsize=12)
        plt.xlabel("Tabula Building Codes", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self.directory_path
            + f"\\Evaluation_{self.evaluation_of}\\All_types_{evaluation}.png"
        )
        plt.close()
        return

    def make_swarm_boxplot(self, data: Any, evaluation: str):

        fig = plt.figure(figsize=(8, 8))
        seaborn.boxplot(data=data)
        seaborn.swarmplot(data=data, color="grey", size=3)
        plt.axhline(y=1, color="red")
        plt.xlabel("Tabula DE buildings", fontsize=12)
        plt.ylabel(f"Ratio of {evaluation} given by HiSim/Tabula", fontsize=12)
        textstr = "\n".join(
            (
                f"Min Ratio {np.round(self.min_ratio,2)}",
                f"Max Ratio {np.round(self.max_ratio,2)}",
                f"Mean Ratio {np.round(self.mean,2)}",
                f"Median Ratio {np.round(self.median,2)}",
                f"Std Ratio {np.round(self.std,2)}"

            )
        )
        plt.text(
            0.2,
            0.95,
            textstr,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        plt.savefig(
            self.directory_path
            + f"\\Evaluation_{self.evaluation_of}\\Swarmplot_Boxplot_all_types_on_{evaluation}.png"
        )
        plt.close()


def main():
    a = TestBuildingHeatingDemandEvaluation(directory_path=directory, file_name=file)


if __name__ == "__main__":
    main()
