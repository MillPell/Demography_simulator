# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:37:49 2023



Please refer to the specification document
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import matplotlib.animation as animation

import yaml

class Age_Pyr_Simu:
    
    
    #Class variable definition
    death_rates = []
    histo_df = pd.DataFrame()
    fig = plt.figure(figsize=(12, 8))
    
    with open('AgeStructureScript_config.yml', 'r') as file:
        conf = yaml.safe_load(file)
    
    # Import Data from .yaml file
    Country = conf['Country']

    #Date option
    start_year = conf['start_year']
    end_year = conf['end_year']

    #Birth rate data
    birth_rate_option = conf['birth_rate_option']
    birth_rate = conf['birth_rate']
    birth_rate_string = conf['birth_rate_string']
    birth_rate_evol_coef = conf['birth_rate_evol_coef']


    #Death rate data
    death_rate_coef_65 = conf['death_rate_coef_65']
    death_rate_coef_85 = conf['death_rate_coef_85']
    death_rate_99 = conf['death_rate_99']
    death_rate_100 = conf['death_rate_100']
    
    #Emigration rate data
    emigration_rate = conf['emigration_rate']

    #Graph option
    histo_bin = conf['histo_bin']    
    graph_bool = conf['graph_bool']

    
    
    
    def excel_to_csv_converter_pop(self):
        """
        Open the excel file from UN :https://population.un.org/wpp/Download/Standard/Population/

        Cut the unnecessary rows and tabs


        Create a csv file with the remaining data. The csv file is way faster to read
    
        This function should be used only once to extract data from the raw excel file. It takes several minutes
        """
        read_file = pd.read_excel('Source_Data/WPP2022_POP_F01_1_POPULATION_SINGLE_AGE_BOTH_SEXES.xlsx', \
                              sheet_name='Estimates', skiprows=16)  

        read_file.to_csv ('Population_data_file.csv', index = None, header=True)
        
    def excel_to_csv_converter_birth_rate(self):
        """
        Open the excel file from UN :https://population.un.org/wpp/Download/Standard/Fertility/

        Cut the unnecessary rows and tabs


        Create a csv file with the remaining data. The csv file is way faster to read
    
        This function should be used only once to extract data from the raw excel file. It takes several minutes
        """
        read_file = pd.read_excel('Source_Data/WPP2022_FERT_F01_FERTILITY_RATES_BY_SINGLE_AGE_OF_MOTHER.xlsx', \
                              sheet_name='Estimates', skiprows=16)  

        read_file.to_csv ('birth_rate_per_age_data_file.csv', index = None, header=True)

    def first_dataframe_creation(self, Country, start_year):
        """
        Extract data from the csv file
    
        Create the Age structure dataframe for the selected start year and country
        """
    
        #Recupération du csv
        base_data = pd.read_csv("Population_data_file.csv")
    
        #Tri par année de départ et Pays
        base_data = base_data[base_data["Region, subregion, country or area *"] == self.Country]
        base_data =  base_data[base_data["Year"] == self.start_year]
    
        #Nettoyage des colonne inutiles
        base_data = base_data.drop(columns=["Index","Variant","Region, subregion, country or area *","Notes","Location code",
                             "ISO3 Alpha-code","ISO2 Alpha-code","SDMX code**","Type","Parent code","Year"])
    

        #Transpose data frame and change the column name
        base_data = base_data.transpose()
    
        base_data = base_data.rename(columns={base_data.columns[0]: str(self.start_year)})
    
    
        #Set create a new column 'age' set to integer
        base_data = base_data.rename(index={'100+': '100'})
        base_data['Age'] = base_data.index

        base_data = base_data.astype({'Age':'int'})
        base_data = base_data.astype({str(self.start_year):'float'})
    
    
        #Convert data a rearrange columns
        base_data[str(self.start_year)] = base_data[str(self.start_year)]\
            .map(lambda pop: round(pop*1000))
        base_data = base_data[['Age',str(self.start_year)]]                      
    
        
        #Creation of a dataframe containing the population in age bins
        low_val = []
        high_val = []
        for i in range(0, len(self.histo_bin)):
            low_val +=re.findall("^(\d+).", self.histo_bin[i])
            high_val +=re.findall("[0-9.]+$", self.histo_bin[i])
    
        high_val += ['101']
   
        self.histo_df = pd.DataFrame({'bins' : self.histo_bin, 'low_val' : low_val, 'high_val' : high_val})
    
    
    
    
        return base_data,self.histo_df
    
    
    
    
    def compute_birth(self, year, pop_per_age_df):

    
        #compute the female population from 15 to 49 years old
        female_fertile_df = pop_per_age_df[['Age',str(year)]].copy()
        female_fertile_df['female pop'] = female_fertile_df[str(year)]\
            .map(lambda pop: pop/2)

    
        female_fertile_df = female_fertile_df[(15 <= pop_per_age_df.Age) & (pop_per_age_df.Age <= 49)]
    
   #Choose the birth rate according to the option
        
        if self.birth_rate_option == 'Fixed':
            current_birth_rate = self.birth_rate
    
            #Get the number of birth per age and total
            births_per_age = female_fertile_df['female pop']\
            .map(lambda pop: round(current_birth_rate*pop/34))
        elif self.birth_rate_option == 'Custom':
            print('Not Yet implemented!!!')
            current_birth_rate = self.birth_rate
    
            #Get the number of birth per age and total
            births_per_age = female_fertile_df['female pop']\
            .map(lambda pop: round(current_birth_rate*pop/34))
            
        elif self.birth_rate_option == 'Fixed_age_specific':
            
            #Recupération du csv
            birth_data = pd.read_csv("birth_rate_per_age_data_file.csv")
    
           #Tri par année de départ et Pays
            birth_data = birth_data[birth_data["Region, subregion, country or area *"] == self.Country]
            birth_data =  birth_data[birth_data["Year"] == self.start_year]
    
            #Nettoyage des colonne inutiles
            birth_data = birth_data.drop(columns=["Index","Variant","Region, subregion, country or area *","Notes","Location code",
                             "ISO3 Alpha-code","ISO2 Alpha-code","SDMX code**","Type","Parent code","Year"])
            
            #Transpose data frame ,set it to integer and change the column name
            birth_data = birth_data.transpose()
            birth_data = birth_data.rename(columns={birth_data.columns[0]: 'birth_rates'})
            birth_data = birth_data.astype({'birth_rates':'float'})

            
            births_per_age = round(female_fertile_df['female pop']*birth_data['birth_rates']/1000)
        
        #Computation of the total birth of the year with the birth year evolution coefficient
        births_total = births_per_age.sum()*np.power(1+self.birth_rate_evol_coef,year-self.start_year)
    
        print ('yearly births: '+ str(births_total))
        print('birth rate evolution coef: ' + str(np.power(1+self.birth_rate_evol_coef,year-self.start_year)))
        return births_total
    
    def compute_death(self, year, pop_per_age_df):
    
        deaths_df = pop_per_age_df.copy()
        

        deaths_df['Death_rates'] = self.death_rates
    
        #Computation of death for each age
        deaths_df['Deaths'] = round(deaths_df['Death_rates']*deaths_df[str(year)])
    
        total_deaths = deaths_df['Deaths'].sum()  
    

        print('yearly deaths: ' + str(total_deaths))
    
        return deaths_df

    def compute_death_rates(self):
    
        #Get the Data from the different .csv from world bank
        #Life_expectancy
        life_expectancy_df = pd.read_csv("Source_Data/WB_esperance_vie.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        life_expectancy = life_expectancy_df.loc[life_expectancy_df['Country Name'] == self.Country, '2020'].values[0]
        #Infant mortality
        mortalite_infant_df = pd.read_csv("Source_Data/WB_mortalite_infant.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        mortalite_infant = mortalite_infant_df.loc[mortalite_infant_df['Country Name'] == self.Country, '2020'].values[0]/1000
        # -5ans  mortality
        mortalite_05ans_df = pd.read_csv("Source_Data/WB_mortalite_05ans.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        mortalite_05ans = mortalite_05ans_df.loc[mortalite_05ans_df['Country Name'] == self.Country, '2020'].values[0]/1000
        # 5-9ans mortality
        mortalite_05_09ans_df = pd.read_csv("Source_Data/WB_mortalite_05_09ans.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        mortalite_05_09ans = mortalite_05_09ans_df.loc[mortalite_05_09ans_df['Country Name'] == self.Country, '2020'].values[0]/1000
        # 10-14ans mortality
        mortalite_10_14ans_df = pd.read_csv("Source_Data/WB_mortalite_10_14ans.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        mortalite_10_14ans = mortalite_10_14ans_df.loc[mortalite_10_14ans_df['Country Name'] == self.Country, '2020'].values[0]/1000
        # 15-19ans mortality
        mortalite_15_19ans_df = pd.read_csv("Source_Data/WB_mortalite_15_19ans.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        mortalite_15_19ans = mortalite_15_19ans_df.loc[mortalite_15_19ans_df['Country Name'] == self.Country, '2020'].values[0]/1000
        # 20-24ans mortality
        mortalite_20_24ans_df = pd.read_csv("Source_Data/WB_mortalite_20_24ans.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        mortalite_20_24ans = mortalite_20_24ans_df.loc[mortalite_20_24ans_df['Country Name'] == self.Country, '2020'].values[0]/1000
        # 65ans homme survivabilite
        survivabilite_homme_65ans_df = pd.read_csv("Source_Data/WB_survivabilite_homme_65ans.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        survivabilite_homme_65ans = survivabilite_homme_65ans_df.loc[survivabilite_homme_65ans_df['Country Name'] == self.Country, '2020'].values[0]/100
        # 65ans femme survivabilite
        survivabilite_femme_65ans_df = pd.read_csv("Source_Data/WB_survivabilite_femme_65ans.csv",usecols = ['Country Name', '2020'] ,skiprows=3)
        survivabilite_femme_65ans = survivabilite_femme_65ans_df.loc[survivabilite_femme_65ans_df['Country Name'] == self.Country, '2020'].values[0]/100
        
        
        #Death rates computation
        
        #infant
        self.death_rates = [mortalite_infant]
        
        #1-4 years old
        mortalite_1_5ans = 1 - (1 - mortalite_05ans)/(1-mortalite_infant)
        for i in range(4):
            self.death_rates.append(1 - np.power(1 - mortalite_1_5ans,1/4))
            
            #5- 24 years old
        for i in range(5):
            self.death_rates.append(1 - np.power(1 - mortalite_05_09ans,1/5))
        for i in range(5):
            self.death_rates.append(1 - np.power(1 - mortalite_10_14ans,1/5))        
        for i in range(5):
            self.death_rates.append(1 - np.power(1 - mortalite_15_19ans,1/5))
        for i in range(5):
            self.death_rates.append(1 - np.power(1 - mortalite_20_24ans,1/5))

        #25- 64 years old
        survivabilite_65ans = (survivabilite_homme_65ans+survivabilite_femme_65ans)/2
        mortalite_25_65ans =1 - survivabilite_65ans/((1 - mortalite_20_24ans)\
                                                     *(1 - mortalite_15_19ans)*(1 - mortalite_10_14ans)\
                                                         *(1 - mortalite_05_09ans)*(1 - mortalite_05ans))
        for i in range(40):
            self.death_rates.append((1+(i-19.5)/40)*(1 - np.power(1 - mortalite_25_65ans,1/40)))
        
        #65- 84 years old  (lineairement de 2* racine40 de  mortalite_25_65ans à 25 fois (originalement 20 fois* )  

        for i in range(20):
            self.death_rates.append((self.death_rate_coef_65+i/19*(self.death_rate_coef_85-self.death_rate_coef_65))*(1 - np.power(1 - mortalite_25_65ans,1/40)))
    
    
        #85 - 99 years old (lineairement de 25 (originalement 20*) racine40 de  mortalite_25_65ans à 0.35 (original: 0.2 )
        
        for i in range(15):
            self.death_rates.append(self.death_rate_coef_85*(1 - np.power(1 - mortalite_25_65ans,1/40))*(1-i/15)+i*(self.death_rate_99)/15)
    
        #100+ years old 0.45 (original: 0.25)
        self.death_rates.append(self.death_rate_100)
    
        if self.graph_bool == True:
            print('graphy')
            death_rate_graph_df = pd.DataFrame(self.death_rates, columns =['Death_rates'])
        
            #graph_init
            ax1 = plt.subplot()
        
        
            #Death graph
            ax1.plot(death_rate_graph_df.index , death_rate_graph_df['Death_rates'], linewidth=1.5, marker='',alpha = 0.75)
        
            plt.ylabel('Death rates (for one thousand people)', color = 'Black', alpha = 0.75)
            plt.xlabel('Age', color = 'Black', alpha = 0.75)
 
        
            #First graph Title
            plt.title('Death rates per age in ' + self.Country,fontsize = 16, color = 'Black', alpha = 0.75) 
        
            # remove the frame of the chart
            for spine in ax1.spines.values():
                spine.set_visible(False)
            
            plt.savefig('Results_files/Death_rates_graph.png')

        return self.death_rates
    
    def next_year_dataframe(self, year,age_repartition_df,histo_df):

        #Use of function to compute the number of birth (age 0 next year)
        yearly_births = self.compute_birth(year, age_repartition_df)
    
    
        #Use of function to compute a df of number of deaths by each age (substract to next year results) 
        yearly_deaths_df = self.compute_death(year, age_repartition_df)    
    
        #Create the column for next year population
        next_year_pop = (age_repartition_df[str(year)]- yearly_deaths_df['Deaths']).values.tolist()
    
        #2022 0 years old are 2021 newborns
        next_year_pop.insert(0, yearly_births)
    
        #2022 new 100 years old are added to 2021 survivors
        next_year_pop[-2] = next_year_pop[-1] + next_year_pop[-2]
        next_year_pop = next_year_pop[:-1]
        
        
        #Emigration rate is added
        next_year_pop = list(map(lambda pop:pop*(1+self.emigration_rate),next_year_pop))
    
        #New column
        age_repartition_df[str(year + 1)] = next_year_pop 
    
    
        #Update of the histo_df used to plot the graphs
        self.histo_df["bin_values_" + str(year)] = 0
        self.histo_df = self.histo_df.astype({'low_val':'int','high_val':'int'})
    
        for j in range(0, len(age_repartition_df.index)):
            for k in range(0, len(self.histo_df.index)):

                if int(age_repartition_df['Age'].values[j]) >= self.histo_df['low_val'].values[k] and int(age_repartition_df['Age'].values[j]) <= self.histo_df['high_val'].values[k]:
                    self.histo_df['bin_values_' + str(year)].values[k] += age_repartition_df[str(year)].values[j]
                
        self.histo_df['bin_values_' + str(year)] = self.histo_df['bin_values_' + str(year)].map(lambda pop: pop/1000000)    
        
        return self.histo_df

    def plot_graphs(self, frame):    
              
        ax = plt.axes()
        
        #Plot the population pyramid graph
        current_year = frame + self.start_year
        #Clear current plot
        plt.cla()
    

        ax.barh(self.histo_df["bins"], self.histo_df["bin_values_" + str(current_year)],color = 'indianred' , edgecolor="white", linewidth=0.7)
    
    
        #Compute the max value of histo_dif as max value for the y axis
        max_histo_df = self.histo_df.copy()
        max_histo_df = max_histo_df.drop(columns = ['bins', 'low_val', 'high_val'])
        max_xaxis = round(max(max_histo_df.max(numeric_only=True))+1)
        plt.xlim(0.,max_xaxis)
    
        #Axis 
        plt.ylabel('Age', color = 'Black', alpha = 0.75, fontsize=18)
        plt.xlabel('million people', color = 'Black', alpha = 0.75, fontsize=18)
        plt.yticks(fontsize=16) 
        plt.xticks(fontsize=16)
    
        #First graph Title
        plt.title('Population pyramid in ' + self.Country,fontsize = 22, color = 'Black', alpha = 0.75,  fontweight="bold") 
        
    
        #Textbox
        plt.text(0.9, 0.9,  'Year: ' + str(current_year), fontsize=18, color='black'\
                 ,ha='center', va='center', alpha = 0.75, transform=ax.transAxes)
        tot_pop = round(self.histo_df["bin_values_" +str(current_year)].sum(),2)
        plt.text(0.9, 0.85,  'Population: ' + str(tot_pop)\
             + ' millions', fontsize=18, color='black',ha='center', \
                 va='center', alpha = 0.75, transform=ax.transAxes)
        
        
        
            # remove the frame of the chart
        for spine in ax.spines.values():
            spine.set_visible(False)
    
        return plt


        




    def main(self):
        """
        

        Main function  

        """
        
        Final_age_repartition_df , self.histo_df= self.first_dataframe_creation(self.Country, self.start_year)

        #compute the death rates data 
        death_rates = self.compute_death_rates()
        
        for i in range(self.start_year, self.end_year):
    
            print('Year: ' + str(i))
    
            #update the dataframe for this year
            self.histo_df = self.next_year_dataframe(i, Final_age_repartition_df, self.histo_df)
        
            print('Population: ' + str(Final_age_repartition_df[str(i)].sum()))
    
        self.histo_df.to_csv ('Results_files/Results_data_file_per_agegroup.csv', index = None, header=True)
        Final_age_repartition_df.to_csv ('Results_files/Results_data_file.csv', index = None, header=True)

        if self.graph_bool:
    
            
    
            a = animation.FuncAnimation(self.fig, self.plot_graphs, frames= self.end_year - self.start_year, interval = 20)
            print('yeah')
            a.save('Results_files/Age_pyramid.gif', writer='writergif', fps=3)
             

if __name__ == "__main__":       
    Age_Pyr_Simu().main()
    
    
