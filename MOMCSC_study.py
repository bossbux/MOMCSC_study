#  Water.java
#
#  Author:
#       Antonio J. Nebro <antonio@lcc.uma.es>
#       Juan J. Durillo <durillo@lcc.uma.es>
#
#  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with self program.  If not, see <http:#www.gnu.org/licenses/>
import sys
import numpy as np
from pymoo.factory import get_problem
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import autograd.numpy as anp
from pymoo.core.problem import Problem
import time
from glob import glob
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import os
import random
from pymoo.vendor import hv
from pymoo.factory import get_performance_indicator
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_problem, get_reference_directions
from pymoo.vendor.hv import HyperVolume
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
np.set_printoptions(threshold=sys.maxsize)


class Service():
#   """ :param number_of_variables: number of decision variables of the problem."""
    def __init__(self, id,
             cloud_id,
             servie_global_id,
             energy,
             cost ,
             reliabelity ,
             availability,
             response_time ,
             benifit):


        self.id = id
        self.energy = energy
        self.cloud_id = cloud_id
        self.servie_global_id = servie_global_id
        self.cost = cost
        self.reliabelity = reliabelity
        self.availability = availability
        self.response_time = response_time
        self.benifit = benifit

    def __repr__(self):
        text = " id " + str(id)+"\n" + \
                " cloud_id " +str(self.cloud_id) +"\n" +\
                " servie_global_id " +str(self.servie_global_id) +"\n" +\
                " energy " +str(self.energy)+"\n" +\
                " cost " +str(self.cost) +"\n" +\
                " reliabelity " +str(self.reliabelity) +"\n" +\
                " availability " +str(self.availability) +"\n" +\
                " response_time " + str(self.response_time)+"\n" +\
                " benifit " + str(self.benifit)
        return text
class Cloud:
#        """ :param number_of_variables: number of decision variables of the problem."""

    cloud_id = 0

    def __init__(self,cloud_id):
        self.cloud_id = cloud_id

        self.services_list = list()
        self.available_composition = {}
    def add_service(self, service_to_add):self.services_list.append(service_to_add)

    def get_services(self): return self.services_list

    def add_service_composition_plan(self,composition_id, service_list_to_add ):
        self.available_composition[composition_id] = service_list_to_add


    def add_service_to_composition_plan(self,  composition_id,  service_id):
        self.available_composition.get(composition_id).add(service_id)

    def get_composition_plan_ids(self, service_id):
        composition_plan_list = list()
        # print('self.available_composition', self.available_composition)
        # print('service_id',service_id)
        for  compsition_id in self.available_composition:

            if(str(service_id) in self.available_composition.get(compsition_id) ):
                composition_plan_list.append(compsition_id)


        return composition_plan_list


    def __repr__(self):
        toPr = "Service List \n"
        toPr += str(self.services_list) + "\n"

        toPr += "Compositions List \n"
        for key in self.available_composition:
            value = self.available_composition.get(key)
            toPr += str(key) + " " + str(value)+ "\n"
        return toPr





class multi_cloud_simulation :
    global_service_id = 0
    global_composition_id = 0
    def __init__(self, path_data_set):
        self.cloud_folder = path_data_set
        #self.cloud_folder = "C:/Users/ahmed.zebouchi/Desktop/JMETALHOME/MOMCSC_experiments/New folder_100c_1000s"
        self.MC_list = list()
        self.services_list = list()
        self.List_files = list()
        self.local_id_and_globale_ids = {}
        self.min_values = np.array([0.0,0.0,0.0,0.0,0.0])
        self.max_values = np.array([0.0,0.0,0.0,0.0,0.0])

    def multi_cloud_simulation(self):

        self.min_values.fill( sys.float_info.max)
        self.max_values.fill( sys.float_info.min)


#        """ :param number_of_variables: number of decision variables of the problem.
#        """

    def read_cloud_providers_services(self):

        list_clouds =  list()



        self.List_files = [f for f in listdir(self.cloud_folder) if isfile(join(self.cloud_folder, f))]
        for file_name in self.List_files :
#            System.out.prln(cloud_folder + file_name)

            tree = ET.parse(self.cloud_folder+file_name)
            root = tree.getroot()
            #print(root.tag, root.attrib)

            cloud_id = root.attrib['id']
            #                System.out.prln("cloud_id "+cloud_id)

            cloud = Cloud(cloud_id)
            services_List = root[0]
                #.getElementsByTagName("service")
            for  service in services_List :
                    service_id = int(service.attrib["id"])

                    node = service.find('energy')
                    service_energy = float(node.text)

                    if service_energy < self.min_values[0]: self.min_values[0] = service_energy
                    if service_energy > self.max_values[0]: self.max_values[0] = service_energy

                    node = service.find('cost')
                    cost = float(node.text)
                    if cost < self.min_values[1]: self.min_values[1] = cost
                    if cost > self.max_values[1]: self.max_values[1] = cost

                    node = service.find('reliability')
                    reliability = float(node.text)
                    if reliability < self.min_values[2]: self.min_values[2] = reliability
                    if reliability > self.max_values[2]: self.max_values[2] = reliability

                    node = service.find('availability')
                    availability = float(node.text)
                    if availability < self.min_values[2]: self.min_values[2] = availability
                    if availability > self.max_values[2]: self.max_values[2] = availability


                    node = service.find('response_time')
                    response_time = float(node.text)
                    if response_time < self.min_values[3]:self.min_values[3] = response_time
                    if response_time > self.max_values[3]: self.max_values[3] = response_time


                    node = service.find('benifit')
                    benifit = float(node.text)
                    if benifit < self.min_values[4]: self.min_values[4] = benifit
                    if benifit > self.max_values[4]: self.max_values[4] = benifit

                    if service_id not in self.local_id_and_globale_ids :#If the map not contains self key (left part)
                        serviceA = list()
                        serviceA.append(int(multi_cloud_simulation.global_service_id))
                        self.local_id_and_globale_ids[service_id] = serviceA
                    else :
                        self.local_id_and_globale_ids[service_id].append(int(multi_cloud_simulation.global_service_id))

                    service =  Service(service_id,
                            cloud_id,
                            multi_cloud_simulation.global_service_id,
                            service_energy,
                            cost,
                            reliability,
                            availability,
                            response_time,
                            benifit)

                    multi_cloud_simulation.global_service_id += 1
                   # print('multi_cloud_simulation.global_service_id',multi_cloud_simulation.global_service_id)
                    cloud.add_service(service)




                    #services_comosition_List = root.getElementsByTagName("composition_plans")[0].getElementsByTagName("composition")

                    services_comosition_List = root[1]
                    for s_comp in services_comosition_List:
                        composition_plan_list = list()
                        for service_ in s_comp:
                            #composition_plan_list.add(service_.getAttribute("id"))
                            composition_plan_list.append(service_.attrib['id'])
                        cloud.add_service_composition_plan(multi_cloud_simulation.global_composition_id,composition_plan_list)
                        multi_cloud_simulation.global_composition_id += 1

            list_clouds.append(cloud)

        return list_clouds



    def get_services_by_global_id(self, list_clouds, global_id) :
        for cloud in list_clouds :
            for service in cloud.services_list :
                if service.servie_global_id == int(global_id):
                    return service
        return

    def get_cloud_id_by_service_global_id(self,list_clouds,  global_id):
        for cloud in list_clouds :
            for service in cloud.services_list :
                if service.servie_global_id == global_id:
                    return cloud
        return



class MOMCSC(ElementwiseProblem)  :

    def __init__(self,folder, **kwargs):

        self.numberOfVariables_ = 5
        xl = np.empty(self.numberOfVariables_)
        xl.fill(1)
        xu = np.empty(self.numberOfVariables_)
        xu.fill(1)

        ########################################
        #   Gave great results but not always  #
        ########################################
        #self.servicces_query = [20, 21,22,16,10]
        #self.servicces_query = [6, 14,19,2,15]
        #path = "C:/Users/zebou/Documents/MC_min/gen_data/"



        #self.servicces_query = [610, 635, 829, 768, 727]
       # path = "C:/Users/zebou/Desktop/phd/doctorat/research/Service composition/phd/Multi-Objective/NSGAII AND III/data/"
        #path = "C:/Users/zebou/Desktop/phd/JMETALHOME/MOMCSC_experiments/MC_sim/"
        #path = "C:/Users/zebou/Documents/Other_try_dataset/data/" ***** [11, 12,13,14,15]
        #path = "C:/Users/zebou/Documents/Other_try_dataset/NC_20_NS_200/"
        path = folder
        self.cloud_simulator = multi_cloud_simulation(path)
        self.list_clouds = self.cloud_simulator.read_cloud_providers_services()

        population = set()
        #print('self.list_clouds ',self.list_clouds )
        for cloud_ in self.list_clouds :
            population.add(cloud_.cloud_id)
        clouds_to_randomly_select = set(population)
        print('clouds_to_randomly_select',population)
        random_cloud = random.choice(list(clouds_to_randomly_select))
        print('chosen clouds befor chosing', random_cloud)
        the_chonsen_cloud = self.list_clouds[int(random_cloud)]
        service_set = set()
        print('set the_chonsen_cloud',the_chonsen_cloud.cloud_id)

        for service in the_chonsen_cloud.services_list :
            service_set.add(service.id)
        print('the_chonsen_cloud.services_list ',service_set )

        service_query = random.sample(service_set, 5)
        print('service_query',service_query)
        #self.servicces_query = [11, 12,13,14,15]
        self.servicces_query = service_query
        #for cloud in self.list_clouds:
         #   print(cloud.cloud_id)
        for var in range(0, self.numberOfVariables_):
            #print('self.servicces_query',self.servicces_query)
            #print('len',len(self.servicces_query))
            #print('var',var)
            #print('self.cloud_simulator.local_id_and_globale_ids.get(self.servicces_query[var])', self.cloud_simulator.local_id_and_globale_ids.get(self.servicces_query[var]))

            xl[var] = np.min(self.cloud_simulator.local_id_and_globale_ids.get(self.servicces_query[var]))
            xu[var] = np.max(self.cloud_simulator.local_id_and_globale_ids.get(self.servicces_query[var]))
        # for



            """
             for cloud in self.list_clouds :
            print('cloud ID', cloud.cloud_id)
            text2 = ''
            print(len(cloud.services_list))
            i = 0
            for service in  cloud.services_list :
                i +=1

                text2 += 'service Id : '+ str(service.id) + ' '
                text2 += ' have as Globale ID :'+  str(service.servie_global_id) +' '

            print(text2)
            print('i',i)
            """
       # print('xl',xl)
        #print('xu',xu)
        print()
        super().__init__(n_var=5,
                         n_obj=8,
                         n_constr=8,
                         xl =xl,
                         xu = xu
                         )


#     [] servicces_query = {2,3,4,5}
#     [] servicces_query = {907,928,774,916}
#     [] servicces_query = {95,97,94,96} the composition of comparaison algors
#     [] servicces_query = {2,26,37,12} #composition for 10 C
#     [] servicces_query = {1,12,24,26}# for 20 C
# for 30 C
#     [] servicces_query = {527,666,813,670,602}
#    "527"666"813"670"




    def _evaluate(self, x, out, *args, **kwargs):

        f =  np.empty(8)
        f.fill(1.)# 8 functions
        involved_cloudes_ids = set()
        energy_consumption = 0
        global_cost = 0
        response_time = 0
        reliability = 0
        profit = 0
        all_services =0
        found_ids = list()
        acceptable_solution = True
        global_Id_text = ""
        involved_composition_list = list()
#        System.out.prln("numberOfVariables_ "+numberOfVariables_)
        for i in range(0, self.numberOfVariables_ ):
            #            System.out.prln("solution!! :"+solution.getDecisionVariables()[i].getValue())
            service = self.cloud_simulator.get_services_by_global_id(self.list_clouds , int(x[i]))
            if service is None:
            #                System.out.prln("no service for "+solution.getDecisionVariables()[i].getValue())
                acceptable_solution = False
                continue
            global_Id_text += str(service.servie_global_id)+" "
            for k in range(0,len(self.servicces_query) ):
                if service.id == self.servicces_query[k] :
                    found_ids.append(service.id)
                    all_services += 1
            found_ids = list(dict.fromkeys(found_ids))
            if len(found_ids) == len(self.servicces_query):
                acceptable_solution = True
            else:
                acceptable_solution = False

            involved_cloudes_ids.add(service.cloud_id)
            energy_consumption += service.energy
            global_cost += service.cost
            response_time += service.response_time
            reliability += service.reliabelity
            # print('cloud ID',self.cloud_simulator.get_cloud_id_by_service_global_id(self.list_clouds, int(x[i])))
            involved_composition_list.extend(self.cloud_simulator.get_cloud_id_by_service_global_id(self.list_clouds, int(x[i])).get_composition_plan_ids(service.id))
            # print('involved_composition_list',involved_composition_list)
            profit+= (service.benifit - service.cost )




        f[0] = energy_consumption
        f[1] = global_cost

        f[2] = -1*reliability
        f[3] = response_time
        f[4] = -1 * profit
        f[5] = len(involved_cloudes_ids)
        frequencymap = {}
        involved_composition_list_ = list(dict.fromkeys(involved_composition_list))
        f[6] = -1* len(involved_composition_list_)
        f[7] = -1* len(found_ids)
        out["F"] = anp.column_stack([f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7]])
        #out["F"] = anp.column_stack(f)
     #   if acceptable_solution:
      #      print("Acceptable")

       #     print('f', f)

      #  print("done ", found_ids)
       # print("len found ", len(found_ids))
        """
                if all_services > 1 :

            #print('el hamdoullah ',f)
            x_= ''
            for i in range(0, len(found_ids)) :
                text = self.cloud_simulator.get_cloud_id_by_service_global_id(self.list_clouds, int(found_ids[i])).cloud_id
                x_ = x_ + ' ' +str(text)
            #print(x_)

        """


        #Once a service not in service list constraint
        #out["F"] = anp.column_stack(f)
        constra = np.empty(self.numberOfVariables_)
        constra.fill(-1)
        for i in range (0, self.numberOfVariables_):
            service = self.cloud_simulator.get_services_by_global_id(self.list_clouds,int(x[i]))
            if(service is not None) :
                if  service.id != self.servicces_query[i] :
                    constra[i] = 1
        #print(constra)
        out["G"] = anp.column_stack(constra)

#} # Water
def get_indicators(pf):
    gd = get_performance_indicator("gd", pf)
    igd = get_performance_indicator("igd", pf)
    gd_plus = get_performance_indicator("gd+", pf)
    igd_plus = get_performance_indicator("igd+", pf)
    hv = get_performance_indicator("hv", pf)
    return gd, igd, gd_plus, igd_plus, hv

########################################################################
# Give the direct path for data set, and the empty folder for results  #
########################################################################
folders = glob("C:/Users/ahmed.zebouchi/Desktop/pymoo/Data/Dataset//*/", recursive = True)
results_folder = 'C:/Users/ahmed.zebouchi/Desktop/pymoo/Results/'
globale_logs_results = results_folder+'global_results.txt'

if not os.path.exists(results_folder+'gd/'):
    os.mkdir(results_folder+'gd/')
if not os.path.exists(results_folder+'igd/'):
    os.mkdir(results_folder+'igd/')
if not os.path.exists(results_folder+'igdplus/'):
    os.mkdir(results_folder+'igdplus/')
if not os.path.exists(results_folder+'gdplus/'):
    os.mkdir(results_folder+'gdplus/')
if not os.path.exists(results_folder+'Exec_time/'):
    os.mkdir(results_folder+'Exec_time/')
    
qos_result_gd = open(results_folder+'gd/globale_logs_results.txt', 'a+')
qos_result_igd = open(results_folder+'igd/globale_logs_results.txt', 'a+')
qos_result_igdplus = open(results_folder+'igdplus/globale_logs_results.txt', 'a+')
qos_result_gdplus = open(results_folder+'gdplus/globale_logs_results.txt', 'a+')
qos_result_Exec_time = open(results_folder+'Exec_time/globale_logs_results.txt', 'a+')
print('folders',folders)

for subfolder in folders:
    print('testing dataset : ', subfolder)
    subfolder += '/*/'
    folder_ = glob(subfolder, recursive=True)
    print('folder_', folder_)
    T_HV = list()
    T_GD = list()
    T_IGD = list()
    T_GD_plus = list()
    T_IGD_plus = list()
    T_Exec_time = list()
    Algos_HV = [None] * 6
    Algos_GD = [None] * 6
    Algos_IGD = [None] * 6
    Algos_GD_plus = [None] * 6
    Algos_IGD_plus = [None] * 6
    Algos_Exec_time = [None] * 6
    print('before loop')
    for folder in folder_:
        print('after loop')
        print('curent dataset : ', folder)
        for i in range(0, 5):
            print('curent execution of: ', folder, ' is: ', i)
            print(folder)
            problem = MOMCSC(folder)
            pth = os.path.join(results_folder, os.path.dirname(folder).split('/')[-1])
            if not os.path.exists(pth.replace('\\', '/')):
                os.makedirs(pth.replace('\\', '/') + '/')
            method = get_algorithm("nsga2",
                                   pop_size=2000,
                                   sampling=get_sampling("int_random"),
                                   crossover=get_crossover("int_sbx"),
                                   mutation=get_mutation("int_pm", eta=3.0),
                                   elimate_duplicates=False,
                                   pf=None,
                                   save_history=False,
                                   verbose=False
                                   )

            print('executing nsga2 pareto calculation...')
            res = minimize(problem,
                           method,
                           termination=('n_gen', 300),
                           seed=1,
                           save_history=True,
                           disp=False
                           )

            # print('table',res.pop.get("F"))
            # print('len(table)',len(res.pop.get("F")))
            print('done executing nsga2 pareto calculation')
            """


            """
            fronts, rank = NonDominatedSorting().do(res.F, return_rank=True)
            non_dominated_front_ref = fronts[0]
            ########################################
            #   Executing NSGAII                   #
            ########################################

            method = get_algorithm("nsga2",
                                   pop_size=2000,
                                   sampling=get_sampling("int_random"),
                                   crossover=get_crossover("int_sbx"),
                                   mutation=get_mutation("int_pm", eta=3.0),
                                   elimate_duplicates=False,
                                   pf=None,
                                   save_history=False,
                                   verbose=False
                                   )
            print('nsga2...')
            res_NSGAII = minimize(problem,
                                  method,
                                  termination=('n_gen', 400),
                                  seed=1,
                                  save_history=True,
                                  disp=False)
            # print('done nsgaII')
            table = res_NSGAII.pop.get("F")
            print('table', table)
            print('len(table)', len(table))
            table_X = res_NSGAII.pop.get("X")
            # Create a numpy array from a list of numbers
            pth = os.path.join(results_folder, os.path.dirname(folder).split('/')[-1])
            if not os.path.exists(pth.replace('\\', '/')):
                os.makedirs(pth.replace('\\', '/'))
            np.savetxt(pth.replace('\\', '/') + '/_res_NSGAII_QoS.out', table, delimiter=',')
            np.savetxt(pth.replace('\\', '/') + '/_res_NSGAII_X.out', table_X, delimiter=',')

            # hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))

            """
            fronts_test, rank = NonDominatedSorting().do(res_NSGAII.F, return_rank=True)

            _front_ref_NSGAII = fronts_test[0]
            max_pop_ref = min(len(_front_ref_NSGAII), len(non_dominated_front_ref))

            _front_ref_NSGAII = np.squeeze(np.asarray(_front_ref_NSGAII))[:max_pop_ref+5]
            non_dominated_front_ref_ = np.squeeze(np.asarray(non_dominated_front_ref))[:max_pop_ref]
            """

            # gd, igd, gd_plus, igd_plus = get_indicators(res.F[non_dominated_front_ref])
            gd, igd, gd_plus, igd_plus, hv = get_indicators(res.F)
            fronts_test, rank = NonDominatedSorting().do(res.F, return_rank=True)

            _front_ref_NSGAII = fronts_test[0]

            NSGAIII_fronts_test, rank = NonDominatedSorting().do(res_NSGAII.F, return_rank=True)

            _front_ref_NSGAIII = NSGAIII_fronts_test[0]
            # print('_front_ref_NSGAII',_front_ref_NSGAII)
            # print('res_NSGAII.F',res_NSGAII.F)
            # print('res.F[_front_ref_NSGAII]',res.F[_front_ref_NSGAII])
            # print('res_NSGAII.F[_front_ref_NSGAIII[0]]',res_NSGAII.F[_front_ref_NSGAIII[0]])
            # print('before fucking it up : ',len(res.F))
            res_copy = np.copy(res.F)
            good_hv = True
            #            res_copy = np.insert(res.F[0], 0,res_NSGAII.F[_front_ref_NSGAIII[0]], 0)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 0] < res_NSGAII.F[_front_ref_NSGAIII[0]][0], :]
            else:
                good_hv = False
            # print('res_copy 0', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 1] < res_NSGAII.F[_front_ref_NSGAIII[0]][1], :]
            else:
                good_hv = False
            # print('res_copy 1', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 2] < res_NSGAII.F[_front_ref_NSGAIII[0]][2], :]

            else:
                good_hv = False
            # print('res_copy 2', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 3] < res_NSGAII.F[_front_ref_NSGAIII[0]][3], :]

            else:
                good_hv = False
            # print('res_copy 3', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 4] < res_NSGAII.F[_front_ref_NSGAIII[0]][4], :]
            else:
                good_hv = False
            # print('res_copy 4', res_copy)

            if len(res_copy):
                X = res_copy[:, [0, 1, 2, 3, 4]]
            else:
                X = None
            # print('res_copy',res_copy)

            # res_copy = res_copy.sort()
            #
            # print('res_copy',res_copy)
            # all_sorted, rank = NonDominatedSorting().do(res_copy, return_rank=True)
            # result = np.where(all_sorted == 0)
            # print('all_sorted',all_sorted)
            # print('result',result)
            # print('all',all)
            # print('res_NSGAII.F[_front_ref_NSGAIII[0]][[0,1,2,3, 4]]',res_NSGAII.F[_front_ref_NSGAIII[0]][[0,1,2,3, 4]])
            hv = get_performance_indicator("hv", res_NSGAII.F[_front_ref_NSGAIII[0]][[0, 1, 2, 3, 4]])
            if not good_hv or X is None or len(X) == 0:
                hv_text = 0
            else:
                hv_text = hv.do(X)

            print('hv_text', hv_text)
            print('after fucking it up : ', len(res.F))
            gd = get_performance_indicator("gd", res.F)
            GD_text = gd.do(res_NSGAII.F)
            print('GD_text', GD_text)
            igd = get_performance_indicator("igd", res.F)
            print('igd', GD_text)
            IGD_text = igd.do(res_NSGAII.F)
            igd_plus = get_performance_indicator("igd+", res.F)
            igd_plus_text = igd_plus.do(res_NSGAII.F)
            print('igd_plus_text', igd_plus_text)
            gd_plus = get_performance_indicator("gd+", res.F)
            gd_plus_text = gd_plus.do(res_NSGAII.F)
            print('gd_plus_text', gd_plus_text)

            qos_results = open(globale_logs_results, 'a+')
            qos_results.write(
                '\n' + os.path.dirname(folder).split('/')[-1] + ' NSGAII hv: ' + str(hv_text) + ' gd: ' + str(
                    GD_text) + ' IGD: ' + str(IGD_text) + ' IGD+: ' + str(igd_plus_text) + ' GD+: ' + str(
                    gd_plus_text) + ' exec time : ' + str(res_NSGAII.exec_time))
            qos_results.close()
            Algos_HV[0] = hv_text
            Algos_GD[0] = GD_text
            Algos_IGD[0] = IGD_text
            Algos_GD_plus[0] = gd_plus_text
            Algos_IGD_plus[0] = igd_plus_text
            Algos_Exec_time[0] = res_NSGAII.exec_time

            Algos_HV[1] = 0
            Algos_GD[1] = 0
            Algos_IGD[1] = 0
            Algos_GD_plus[1] = 0
            Algos_IGD_plus[1] = 0
            Algos_Exec_time[1] = 0
            # try:
            method = get_algorithm("rnsga3",
                                   ref_points=res_NSGAII.F,
                                   pop_per_ref_point=8, sampling=get_sampling("real_random"),
                                   crossover=get_crossover("int_sbx"),
                                   mutation=get_mutation("int_pm", eta=3.0),
                                   parallelization=("threads", 2)
                                   )

            print('rnsga3...')
            res_rnsga3 = minimize(problem,
                                  method,
                                  termination=('n_gen', 300),
                                  seed=1,
                                  save_history=True,
                                  disp=False)
            print('done rnsga3')

            table = res_rnsga3.pop.get("F")
            table_X = res_rnsga3.pop.get("X")

            np.savetxt(pth.replace('\\', '/') + '/_res_rnsga3_QoS.out', table, delimiter=',')
            np.savetxt(pth.replace('\\', '/') + '/_res_rnsga3_X.out', table_X, delimiter=',')
            # Create a numpy array from a list of numbers
            """
            fronts_test, rank = NonDominatedSorting().do(res_rnsga3.F, return_rank=True)

            _front_ref_RNSGAIII = fronts_test[0]

            max_pop_ref = min(len(_front_ref_RNSGAIII), len(non_dominated_front_ref))
            _front_ref_RNSGAIII = np.squeeze(np.asarray(_front_ref_RNSGAIII))[:max_pop_ref]
            non_dominated_front_ref_RNSGAIII = np.squeeze(np.asarray(non_dominated_front_ref))[:max_pop_ref]
            """
            # hv = get_performance_indicator("hv", res.F)
            # hv_text = hv.do(res_rnsga3.F)
            res_rnsga3_fronts_test, rank = NonDominatedSorting().do(res_rnsga3.F, return_rank=True)

            _front_ref_rnsga3 = res_rnsga3_fronts_test[0]
            res_copy = np.copy(res.F)
            good_hv = True
            #            res_copy = np.insert(res.F[0], 0,res_NSGAII.F[_front_ref_NSGAIII[0]], 0)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 0] < res_NSGAII.F[_front_ref_rnsga3[0]][0], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 1] < res_NSGAII.F[_front_ref_rnsga3[0]][1], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 2] < res_NSGAII.F[_front_ref_rnsga3[0]][2], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 3] < res_NSGAII.F[_front_ref_rnsga3[0]][3], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 4] < res_NSGAII.F[_front_ref_rnsga3[0]][4], :]
            else:
                good_hv = False
            if len(res_copy):
                X = res_copy[:, [0, 1, 2, 3, 4]]
            else:
                X = None
            hv = get_performance_indicator("hv", res_NSGAII.F[_front_ref_rnsga3[0]][[0, 1, 2, 3, 4]])
            if not good_hv or X is None or len(X) == 0:
                hv_text = 0
            else:
                hv_text = hv.do(X)

            print('hv_text', hv_text)
            gd = get_performance_indicator("gd", res.F)
            GD_text = gd.do(res_rnsga3.F)
            igd = get_performance_indicator("igd", res.F)
            IGD_text = igd.do(res_rnsga3.F)
            igd_plus = get_performance_indicator("igd+", res.F)
            igd_plus_text = igd_plus.do(res_rnsga3.F)
            gd_plus = get_performance_indicator("gd+", res.F)
            gd_plus_text = gd_plus.do(res_rnsga3.F)

            qos_results = open(globale_logs_results, 'a+')
            qos_results.write(
                '\n' + os.path.dirname(folder).split('/')[-1] + ' RNSGAIII hv: ' + str(hv_text) + ' gd: ' + str(
                    GD_text) + ' IGD: ' + str(IGD_text) + ' IGD+: ' + str(igd_plus_text) + ' GD+: ' + str(
                    gd_plus_text) + ' exec time : ' + str(res_rnsga3.exec_time))
            qos_results.close()

            Algos_HV[1] = hv_text
            Algos_GD[1] = GD_text
            Algos_IGD[1] = IGD_text
            Algos_GD_plus[1] = gd_plus_text
            Algos_IGD_plus[1] = igd_plus_text
            Algos_Exec_time[1] = res_rnsga3.exec_time
            # except Exception as e:
            #     print('the error is in here',e)
            Algos_HV[2] = 0
            Algos_GD[2] = 0
            Algos_IGD[2] = 0
            Algos_GD_plus[2] = 0
            Algos_IGD_plus[2] = 0
            Algos_Exec_time[2] = 0
            try:
                method = get_algorithm("rnsga3_hybrid",
                                       ref_points=res.F,
                                       pop_per_ref_point=8, sampling=get_sampling("real_random"),
                                       crossover=get_crossover("int_sbx"),
                                       mutation=get_mutation("int_pm", eta=3.0),
                                       parallelization=("threads", 2)
                                       )
                print('pRTMNSGAII...')
                res_pRTMNSGAII = minimize(problem,
                                          method,
                                          termination=('n_gen', 10),
                                          seed=1,
                                          save_history=True,
                                          disp=False)

                """
                fronts_test, rank = NonDominatedSorting().do(res_pRTMNSGAII.F, return_rank=True)
                _front_ref_pRTMNSGAII = fronts_test[0]
                max_pop_ref = min(len(_front_ref_pRTMNSGAII), len(non_dominated_front_ref))
                _front_ref_pRTMNSGAII = np.squeeze(np.asarray(_front_ref_pRTMNSGAII))[:max_pop_ref]
                non_dominated_front_ref_pRTMNSGAII = np.squeeze(np.asarray(non_dominated_front_ref))[:max_pop_ref]

                """
                print('done pRTMNSGAII')

                # hv = get_performance_indicator("hv", res.F)
                # hv_text = hv.do(res_pRTMNSGAII.F)
                # hv_text = hv.do(res_pRTMNSGAII.F)

                res_pRTMNSGAII_fronts_test, rank = NonDominatedSorting().do(res_pRTMNSGAII.F, return_rank=True)

                _front_ref_pRTMNSGAII = res_pRTMNSGAII_fronts_test[0]
                res_copy = np.copy(res.F)
                good_hv = True
                #            res_copy = np.insert(res.F[0], 0,res_NSGAII.F[_front_ref_NSGAIII[0]], 0)
                if len(res_copy):
                    res_copy = res_copy[res_copy[:, 0] < res.F[_front_ref_pRTMNSGAII[0]][0], :]
                else:
                    good_hv = False
                if len(res_copy):
                    res_copy = res_copy[res_copy[:, 1] < res.F[_front_ref_pRTMNSGAII[0]][1], :]
                else:
                    good_hv = False
                if len(res_copy):
                    res_copy = res_copy[res_copy[:, 2] < res.F[_front_ref_pRTMNSGAII[0]][2], :]
                else:
                    good_hv = False
                if len(res_copy):
                    res_copy = res_copy[res_copy[:, 3] < res.F[_front_ref_pRTMNSGAII[0]][3], :]
                else:
                    good_hv = False
                if len(res_copy):
                    res_copy = res_copy[res_copy[:, 4] < res.F[_front_ref_pRTMNSGAII[0]][4], :]
                else:
                    good_hv = False
                if len(res_copy):
                    X = res_copy[:, [0, 1, 2, 3, 4]]
                else:
                    X = None
                hv = get_performance_indicator("hv", res.F[_front_ref_pRTMNSGAII[0]][[0, 1, 2, 3, 4]])
                if not good_hv or X is None or len(X) == 0:
                    hv_text = 0
                else:
                    hv_text = hv.do(X)

                gd = get_performance_indicator("gd", res.F)
                GD_text = gd.do(res_pRTMNSGAII.F)
                igd = get_performance_indicator("igd", res.F)
                IGD_text = igd.do(res_pRTMNSGAII.F)
                igd_plus = get_performance_indicator("igd+", res.F)
                igd_plus_text = igd_plus.do(res_pRTMNSGAII.F)
                gd_plus = get_performance_indicator("gd+", res.F)
                gd_plus_text = gd_plus._do(res_pRTMNSGAII.F)
                qos_results = open(globale_logs_results, 'a+')
                qos_results.write(
                    '\n' + os.path.dirname(folder).split('/')[-1] + ' pRTMNSGAII hv: ' + str(hv_text) + ' gd: ' + str(
                        GD_text) + ' IGD: ' + str(IGD_text) + ' IGD+: ' + str(igd_plus_text) + ' GD+: ' + str(
                        gd_plus_text) + ' exec time : ' + str(res_pRTMNSGAII.exec_time))

                qos_results.close()
                Algos_HV[2] = hv_text
                Algos_GD[2] = GD_text
                Algos_IGD[2] = IGD_text
                Algos_GD_plus[2] = gd_plus_text
                Algos_IGD_plus[2] = igd_plus_text
                Algos_Exec_time[2] = res_pRTMNSGAII.exec_time
                table = res_pRTMNSGAII.pop.get("F")
                table_X = res_pRTMNSGAII.pop.get("X")
                # Create a numpy array from a list of numbers

                np.savetxt(pth.replace('\\', '/') + '/_res_pRTMNSGAII_QoS.out', table, delimiter=',')
                np.savetxt(pth.replace('\\', '/') + '/_res_pRTMNSGAII_X.out', table_X, delimiter=',')
            except Exception as e:
                print('error in here too', e)
            Algos_HV[3] = 0
            Algos_GD[3] = 0
            Algos_IGD[3] = 0
            Algos_GD_plus[3] = 0
            Algos_IGD_plus[3] = 0
            Algos_Exec_time[3] = 0
            # try:
            ref_dirs = get_reference_directions("das-dennis", 8, n_partitions=3)
            # print('ref_dirs', ref_dirs)
            method = get_algorithm("TMNSGA3",
                                   ref_dirs=ref_dirs,
                                   pop_per_ref_point=8, sampling=get_sampling("real_random"),
                                   crossover=get_crossover("int_sbx"),
                                   mutation=get_mutation("int_pm", eta=3.0),
                                   parallelization=("threads", 1),
                                   )
            print('tmnsga3...')
            res_tmnsga3 = minimize(problem,
                                   method,
                                   termination=('n_gen', 100),
                                   seed=1,
                                   save_history=True,
                                   disp=False)
            print('done tmnsga3')
            table = res_tmnsga3.pop.get("F")
            table_X = res_tmnsga3.pop.get("X")
            np.savetxt(pth.replace('\\', '/') + '/_res_tmnsga3.out', table, delimiter=',')
            np.savetxt(pth.replace('\\', '/') + '/_res_tmnsga3.out', table_X, delimiter=',')
            # Create a numpy array from a list of numbers
            """
            fronts_test, rank = NonDominatedSorting().do(res_tmnsga3.F, return_rank=True)
            _front_ref_tmnsga3 = fronts_test[0]
            max_pop_ref = min(len(_front_ref_tmnsga3), len(non_dominated_front_ref))
            _front_ref_tmnsga3 = np.squeeze(np.asarray(_front_ref_tmnsga3))[:max_pop_ref]
            non_dominated_front_ref_tmnsga3 = np.squeeze(np.asarray(non_dominated_front_ref))[:max_pop_ref]
            """
            # hv = get_performance_indicator("hv", res.F)
            # hv_text = hv.do(res_tmnsga3.F)

            res_tmnsga3_fronts_test, rank = NonDominatedSorting().do(res_tmnsga3.F, return_rank=True)

            _front_ref_tmnsga3 = res_tmnsga3_fronts_test[0]
            res_copy = np.copy(res.F)
            good_hv = True
            #            res_copy = np.insert(res.F[0], 0,res_NSGAII.F[_front_ref_NSGAIII[0]], 0)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 0] < res.F[_front_ref_tmnsga3[0]][0], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 1] < res.F[_front_ref_tmnsga3[0]][1], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 2] < res.F[_front_ref_tmnsga3[0]][2], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 3] < res.F[_front_ref_tmnsga3[0]][3], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 4] < res.F[_front_ref_tmnsga3[0]][4], :]
            else:
                good_hv = False
            if len(res_copy):
                X = res_copy[:, [0, 1, 2, 3, 4]]
            else:
                X = None
            hv = get_performance_indicator("hv", res.F[_front_ref_tmnsga3[0]][[0, 1, 2, 3, 4]])
            if not good_hv or X is None or len(X) == 0:
                hv_text = 0
            else:
                hv_text = hv.do(X)

            gd = get_performance_indicator("gd", res.F)
            GD_text = gd.do(res_tmnsga3.F)
            igd = get_performance_indicator("igd", res.F)
            IGD_text = igd.do(res_tmnsga3.F)
            igd_plus = get_performance_indicator("igd+", res.F)
            igd_plus_text = igd_plus.do(res_tmnsga3.F)
            gd_plus = get_performance_indicator("gd+", res.F)
            gd_plus_text = gd_plus.do(res_tmnsga3.F)

            qos_results = open(globale_logs_results, 'a+')
            qos_results.write(
                '\n' + os.path.dirname(folder).split('/')[-1] + ' TMNSGAIII hv: ' + str(hv_text) + ' gd: ' + str(
                    GD_text) + ' IGD: ' + str(IGD_text) + ' IGD+: ' + str(igd_plus_text) + ' GD+: ' + str(
                    gd_plus_text) + ' exec time : ' + str(res_tmnsga3.exec_time))
            qos_results.close()

            Algos_HV[3] = hv_text
            Algos_GD[3] = GD_text
            Algos_IGD[3] = IGD_text
            Algos_GD_plus[3] = gd_plus_text
            Algos_IGD_plus[3] = igd_plus_text
            Algos_Exec_time[3] = res_tmnsga3.exec_time

            # except Exception as e:
            #     print('res_tmnsga3 problem',e)
            T_HV.append(Algos_HV)
            T_GD.append(Algos_GD)
            T_IGD.append(Algos_IGD)
            T_GD_plus.append(Algos_GD_plus)
            T_IGD_plus.append(Algos_IGD_plus)
            T_Exec_time.append(Algos_Exec_time)

            Algos_HV[4] = 0
            Algos_GD[4] = 0
            Algos_IGD[4] = 0
            Algos_GD_plus[4] = 0
            Algos_IGD_plus[4] = 0
            Algos_Exec_time[4] = 0
            # try:
            ref_dirs = get_reference_directions("das-dennis", 8, n_partitions=3)
            # print('ref_dirs', ref_dirs)
            method = get_algorithm("nsga3",
                                   ref_dirs=ref_dirs,
                                   pop_per_ref_point=8, sampling=get_sampling("real_random"),
                                   crossover=get_crossover("int_sbx"),
                                   mutation=get_mutation("int_pm", eta=3.0),
                                   parallelization=("threads", 1),
                                   )
            print('NSGAIII...')
            res_nsga3 = minimize(problem,
                                   method,
                                   termination=('n_gen', 100),
                                   seed=1,
                                   save_history=True,
                                   disp=False)
            print('done tmnsga3')
            table = res_tmnsga3.pop.get("F")
            table_X = res_tmnsga3.pop.get("X")
            np.savetxt(pth.replace('\\', '/') + '/_res_nsga3.out', table, delimiter=',')
            np.savetxt(pth.replace('\\', '/') + '/_res_nsga3.out', table_X, delimiter=',')
            res_tmnsga3_fronts_test, rank = NonDominatedSorting().do(res_nsga3.F, return_rank=True)

            _front_ref_nsga3 = res_tmnsga3_fronts_test[0]
            res_copy = np.copy(res.F)
            good_hv = True
            #            res_copy = np.insert(res.F[0], 0,res_NSGAII.F[_front_ref_NSGAIII[0]], 0)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 0] < res.F[_front_ref_nsga3[0]][0], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 1] < res.F[_front_ref_nsga3[0]][1], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 2] < res.F[_front_ref_nsga3[0]][2], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 3] < res.F[_front_ref_nsga3[0]][3], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 4] < res.F[_front_ref_nsga3[0]][4], :]
            else:
                good_hv = False
            if len(res_copy):
                X = res_copy[:, [0, 1, 2, 3, 4]]
            else:
                X = None
            hv = get_performance_indicator("hv", res.F[_front_ref_nsga3[0]][[0, 1, 2, 3, 4]])
            if not good_hv or X is None or len(X) == 0:
                hv_text = 0
            else:
                hv_text = hv.do(X)

            gd = get_performance_indicator("gd", res.F)
            GD_text = gd.do(res_tmnsga3.F)
            igd = get_performance_indicator("igd", res.F)
            IGD_text = igd.do(res_tmnsga3.F)
            igd_plus = get_performance_indicator("igd+", res.F)
            igd_plus_text = igd_plus.do(res_tmnsga3.F)
            gd_plus = get_performance_indicator("gd+", res.F)
            gd_plus_text = gd_plus.do(res_tmnsga3.F)

            qos_results = open(globale_logs_results, 'a+')
            qos_results.write(
                '\n' + os.path.dirname(folder).split('/')[-1] + ' NSGAIII hv: ' + str(hv_text) + ' gd: ' + str(
                    GD_text) + ' IGD: ' + str(IGD_text) + ' IGD+: ' + str(igd_plus_text) + ' GD+: ' + str(
                    gd_plus_text) + ' exec time : ' + str(res_tmnsga3.exec_time))
            qos_results.close()

            Algos_HV[4] = hv_text
            Algos_GD[4] = GD_text
            Algos_IGD[4] = IGD_text
            Algos_GD_plus[4] = gd_plus_text
            Algos_IGD_plus[4] = igd_plus_text
            Algos_Exec_time[4] = res_tmnsga3.exec_time

            # except Exception as e:
            #     print('res_tmnsga3 problem',e)
            T_HV.append(Algos_HV)
            T_GD.append(Algos_GD)
            T_IGD.append(Algos_IGD)
            T_GD_plus.append(Algos_GD_plus)
            T_IGD_plus.append(Algos_IGD_plus)
            T_Exec_time.append(Algos_Exec_time)

            """
            pTMSGNAII
            """

            method = get_algorithm("nsga2",
                                   pop_size=2000,
                                   sampling=get_sampling("int_random"),
                                   crossover=get_crossover("int_sbx"),
                                   mutation=get_mutation("int_pm", eta=3.0),
                                   elimate_duplicates=False,
                                   pf=None,
                                   save_history=False,
                                   parallelization=("threads", 4),
                                   verbose=False
                                   )
            print('nsga2...')
            res_pNSGAII = minimize(problem,
                                   method,
                                   termination=('n_gen', 400),
                                   seed=1,
                                   save_history=True,
                                   disp=False)
            # print('done nsgaII')
            table = res_pNSGAII.pop.get("F")
            print('table', table)
            print('len(table)', len(table))
            table_X = res_pNSGAII.pop.get("X")
            # Create a numpy array from a list of numbers
            pth = os.path.join(results_folder, os.path.dirname(folder).split('/')[-1])
            if not os.path.exists(pth.replace('\\', '/')):
                os.makedirs(pth.replace('\\', '/'))
            np.savetxt(pth.replace('\\', '/') + '/_res_NSGAII_QoS.out', table, delimiter=',')
            np.savetxt(pth.replace('\\', '/') + '/_res_NSGAII_X.out', table_X, delimiter=',')

            # gd, igd, gd_plus, igd_plus = get_indicators(res.F[non_dominated_front_ref])
            gd, igd, gd_plus, igd_plus, hv = get_indicators(res.F)
            fronts_test, rank = NonDominatedSorting().do(res.F, return_rank=True)

            _front_ref_NSGAII = fronts_test[0]

            pNSGAIII_fronts_test, rank = NonDominatedSorting().do(res_pNSGAII.F, return_rank=True)

            _front_ref_pNSGAIII = pNSGAIII_fronts_test[0]
            res_copy = np.copy(res.F)
            good_hv = True
            #            res_copy = np.insert(res.F[0], 0,res_NSGAII.F[_front_ref_NSGAIII[0]], 0)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 0] < res_pNSGAII.F[_front_ref_pNSGAIII[0]][0], :]
            else:
                good_hv = False
            # print('res_copy 0', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 1] < res_pNSGAII.F[_front_ref_pNSGAIII[0]][1], :]
            else:
                good_hv = False
            # print('res_copy 1', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 2] < res_pNSGAII.F[_front_ref_pNSGAIII[0]][2], :]

            else:
                good_hv = False
            # print('res_copy 2', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 3] < res_pNSGAII.F[_front_ref_pNSGAIII[0]][3], :]

            else:
                good_hv = False
            # print('res_copy 3', res_copy)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 4] < res_pNSGAII.F[_front_ref_pNSGAIII[0]][4], :]
            else:
                good_hv = False
            # print('res_copy 4', res_copy)

            if len(res_copy):
                X = res_copy[:, [0, 1, 2, 3, 4]]
            else:
                X = None
            hv = get_performance_indicator("hv", res_pNSGAII.F[_front_ref_NSGAIII[0]][[0, 1, 2, 3, 4]])
            if not good_hv or X is None or len(X) == 0:
                hv_text = 0
            else:
                hv_text = hv.do(X)

            print('hv_text', hv_text)
            print('after fucking it up : ', len(res.F))
            gd = get_performance_indicator("gd", res.F)
            GD_text = gd.do(res_pNSGAII.F)
            print('GD_text', GD_text)
            igd = get_performance_indicator("igd", res.F)
            print('igd', GD_text)
            IGD_text = igd.do(res_pNSGAII.F)
            igd_plus = get_performance_indicator("igd+", res.F)
            igd_plus_text = igd_plus.do(res_pNSGAII.F)
            print('igd_plus_text', igd_plus_text)
            gd_plus = get_performance_indicator("gd+", res.F)
            gd_plus_text = gd_plus.do(res_pNSGAII.F)
            print('gd_plus_text', gd_plus_text)

            qos_results = open(globale_logs_results, 'a+')
            qos_results.write(
                '\n' + os.path.dirname(folder).split('/')[-1] + ' NSGAII hv: ' + str(hv_text) + ' gd: ' + str(
                    GD_text) + ' IGD: ' + str(IGD_text) + ' IGD+: ' + str(igd_plus_text) + ' GD+: ' + str(
                    gd_plus_text) + ' exec time : ' + str(res_pNSGAII.exec_time))
            qos_results.close()
            Algos_HV[5] = hv_text
            Algos_GD[5] = GD_text
            Algos_IGD[5] = IGD_text
            Algos_GD_plus[5] = gd_plus_text
            Algos_IGD_plus[5] = igd_plus_text
            Algos_Exec_time[5] = res_pNSGAII.exec_time

            T_HV.append(Algos_HV)
            T_GD.append(Algos_GD)
            T_IGD.append(Algos_IGD)
            T_GD_plus.append(Algos_GD_plus)
            T_IGD_plus.append(Algos_IGD_plus)
            T_Exec_time.append(Algos_Exec_time)
            
            
            # print('ref_dirs', ref_dirs)
            method = get_algorithm("nspso", 1000,50)
            print('NSGAIII...')
            res_nspso = minimize(problem,
                                   method,
                                   termination=('n_gen', 100),
                                   seed=1,
                                   save_history=True,
                                   disp=False)
            print('done tmnsga3')
            table = res_tmnsga3.pop.get("F")
            table_X = res_tmnsga3.pop.get("X")
            np.savetxt(pth.replace('\\', '/') + '/_res_nsga3.out', table, delimiter=',')
            np.savetxt(pth.replace('\\', '/') + '/_res_nsga3.out', table_X, delimiter=',')
            res_nspso_fronts_test, rank = NonDominatedSorting().do(res_nspso.F, return_rank=True)

            _front_ref_nspso = res_nspso_fronts_test[0]
            res_copy = np.copy(res.F)
            good_hv = True
            #            res_copy = np.insert(res.F[0], 0,res_NSGAII.F[_front_ref_NSGAIII[0]], 0)
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 0] < res.F[_front_ref_nspso[0]][0], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 1] < res.F[_front_ref_nspso[0]][1], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 2] < res.F[_front_ref_nspso[0]][2], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 3] < res.F[_front_ref_nspso[0]][3], :]
            else:
                good_hv = False
            if len(res_copy):
                res_copy = res_copy[res_copy[:, 4] < res.F[_front_ref_nspso[0]][4], :]
            else:
                good_hv = False
            if len(res_copy):
                X = res_copy[:, [0, 1, 2, 3, 4]]
            else:
                X = None
            hv = get_performance_indicator("hv", res.F[_front_ref_nspso[0]][[0, 1, 2, 3, 4]])
            if not good_hv or X is None or len(X) == 0:
                hv_text = 0
            else:
                hv_text = hv.do(X)

            gd = get_performance_indicator("gd", res.F)
            GD_text = gd.do(res_tmnsga3.F)
            igd = get_performance_indicator("igd", res.F)
            IGD_text = igd.do(res_tmnsga3.F)
            igd_plus = get_performance_indicator("igd+", res.F)
            igd_plus_text = igd_plus.do(res_tmnsga3.F)
            gd_plus = get_performance_indicator("gd+", res.F)
            gd_plus_text = gd_plus.do(res_tmnsga3.F)

            qos_results = open(globale_logs_results, 'a+')
            qos_results.write(
                '\n' + os.path.dirname(folder).split('/')[-1] + ' NSPSO hv: ' + str(hv_text) + ' gd: ' + str(
                    GD_text) + ' IGD: ' + str(IGD_text) + ' IGD+: ' + str(igd_plus_text) + ' GD+: ' + str(
                    gd_plus_text) + ' exec time : ' + str(res_tmnsga3.exec_time))
            qos_results.close()

            Algos_HV[6] = hv_text
            Algos_GD[6] = GD_text
            Algos_IGD[6] = IGD_text
            Algos_GD_plus[6] = gd_plus_text
            Algos_IGD_plus[6] = igd_plus_text
            Algos_Exec_time[6] = res_tmnsga3.exec_time

            # except Exception as e:
            #     print('res_tmnsga3 problem',e)
            T_HV.append(Algos_HV)
            T_GD.append(Algos_GD)
            T_IGD.append(Algos_IGD)
            T_GD_plus.append(Algos_GD_plus)
            T_IGD_plus.append(Algos_IGD_plus)
            T_Exec_time.append(Algos_Exec_time)

            print('========done executing one dataset============')
        print('========done executing dataset============')
    qos_result_hv = open(results_folder + 'hv/' + str(subfolder[:-3].split('\\')[-2]) + '.txt', 'a+')
    text = 'nsga2 , rnsga3, rnsga3_hybrid, tmnsga3, NSGAIII , pnsga2, nspso'
    for elem in T_HV:
        text += '\n' + str(elem[0]) + ',' + str(elem[1]) + ',' + str(elem[2]) + ',' + str( elem[3]) + ',' + str(elem[4]) + ',' + str(elem[5])+ ',' + str(elem[6])
    qos_result_hv.write(text)
    qos_result_hv.close()

    qos_result_gd = open(results_folder + 'gd/' + str(subfolder[:-3].split('\\')[-2]) + 'txt', 'a+')
    text = 'nsga2 , rnsga3, rnsga3_hybrid, tmnsga3, NSGAIII , pnsga2, nspso'
    for elem in T_GD:
        text += '\n' + str(elem[0]) + ',' + str(elem[1]) + ',' + str(elem[2]) + ',' + str(elem[3]) + ',' + str(
            elem[4]) + ',' + str(elem[5])+ ',' + str(elem[6])
    qos_result_gd.write(text)
    qos_result_gd.close()

    qos_result_igd = open(results_folder + 'igd/' + str(subfolder[:-3].split('\\')[-2]) + 'txt', 'a+')
    text = 'nsga2 , rnsga3, rnsga3_hybrid, tmnsga3, NSGAIII , pnsga2, nspso'
    for elem in T_IGD:
        text += '\n' + str(elem[0]) + ',' + str(elem[1]) + ',' + str(elem[2]) + ',' + str(elem[3]) + ',' + str(
            elem[4]) + ',' + str(elem[5])+ ',' + str(elem[6])
    qos_result_igd.write(text)
    qos_result_igd.close()

    qos_result_igdplus = open(results_folder + 'igdplus/' + str(subfolder[:-3].split('\\')[-2]) + 'txt', 'a+')
    text = 'nsga2 , rnsga3, rnsga3_hybrid, tmnsga3, NSGAIII , pnsga2, nspso'
    for elem in T_IGD_plus:
        text += '\n' + str(elem[0]) + ',' + str(elem[1]) + ',' + str(elem[2]) + ',' + str(elem[3]) + ',' + str(
            elem[4]) + ',' + str(elem[5])+ ',' + str(elem[6])
    qos_result_igdplus.write(text)
    qos_result_igdplus.close()

    qos_result_gdplus = open(results_folder + 'gdplus/' + str(subfolder[:-3].split('\\')[-2]) + 'txt', 'a+')
    text = 'nsga2 , rnsga3, rnsga3_hybrid, tmnsga3, NSGAIII , pnsga2, nspso'
    for elem in T_GD_plus:
        text += '\n' + str(elem[0]) + ',' + str(elem[1]) + ',' + str(elem[2]) + ',' + str(elem[3]) + ',' + str(
            elem[4]) + ',' + str(elem[5])+ ',' + str(elem[6])
    qos_result_gdplus.write(text)
    qos_result_gdplus.close()

    qos_result_Exec_time = open(results_folder + 'Exec_time/' + str(subfolder[:-3].split('\\')[-2]) + 'txt', 'a+')
    text = 'nsga2 , rnsga3, rnsga3_hybrid, tmnsga3, NSGAIII , pnsga2, nspso'
    for elem in T_Exec_time:
        text += '\n' + str(elem[0]) + ',' + str(elem[1]) + ',' + str(elem[2]) + ',' + str(elem[3]) + ',' + str(
            elem[4]) + ',' + str(elem[5])+ ',' + str(elem[6])
    qos_result_Exec_time.write(text)
    qos_result_Exec_time.close()

