from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork, JunctionTree, BayesianModel
from pgmpy.inference import BeliefPropagation

'''
TODO 6.1 Oblicz rozkłady brzegowe dla ciągu obserwacji true, true, false, true, true za pomocą algorytmu belief propagation
'''
model = BayesianModel([('Rain_t_1', 'Rain_t'), ('Rain_t', 'Umbrella_t')])

cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, values=[[.4], [.6]])

cpd_rain_t = TabularCPD('Rain_t', variable_card=2,
                       values=[[0.7, 0.3],
                               [0.3, 0.7]], evidence=['Rain_t_1'], evidence_card=[2])
cpd_umbrella_t = TabularCPD('Umbrella_t', variable_card=2,
                                   values=[[0.8, 0.1],
                                           [0.2, 0.9]], evidence=['Rain_t'], evidence_card=[2])

model.add_cpds(cpd_rain_t_1, cpd_rain_t, cpd_umbrella_t)
# Checking if model is correctly added
print('Check model :', model.check_model())
print('Edges: ', model.number_of_edges())


belief_propagation = BeliefPropagation(model)
belief_propagation.calibrate()

kolejne_dni = []
for i in range(5):
    if i == 2:
        q = belief_propagation.query(['Rain_t'], evidence={'Umbrella_t': 0})
        kolejne_dni.append(q.values[1])
    else:
        q = belief_propagation.query(['Rain_t'], evidence={'Umbrella_t': 1})
        kolejne_dni.append(q.values[1])
    cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
    model.add_cpds(cpd_rain_t_1)
print('P(R1 | U1) =\n', q)

# 5.2
print(kolejne_dni)




