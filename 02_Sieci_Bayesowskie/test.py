from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import numpy as np

kolejne_dni = []

G = FactorGraph()
G.add_node('Rain_t_1')
G.add_node('Rain_t')
G.add_node('Umbrella_t')

phi1 = DiscreteFactor(['Rain_t_1'], [2], values=np.array([0.4, 0.6]))
phi2 = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], values=np.array([0.7, 0.3, 0.3, 0.7]))
phi3 = DiscreteFactor(['Umbrella_t', 'Rain_t'], [2, 2], values=np.array([0.8, 0.1, 0.2, 0.9]))

G.add_factors(phi1, phi2, phi3)

# G.add_edges_from([('Rain_t_1', phi1), ('Rain_t', phi2), ('Rain_t_1', phi2), ('Umbrella_t', phi3), ('Rain_t', phi3)])
G.add_edges_from([('Rain_t_1', phi1), ('Rain_t_1', phi2), ('Rain_t', phi2), ('Rain_t', phi3), ('Umbrella_t', phi3)])

# Checking if model is correctly added
print('Check model :', G.check_model())
print(G.number_of_edges())
print(G.get_variable_nodes())
print(G.get_factors(node=phi3))


belief_propagation = BeliefPropagation(G)

# Dzień 0 - predykcja wartości dla Rain_t_1
q = belief_propagation.query(['Rain_t_1'])
print('Dla dnia 0 =\n', q)
kolejne_dni.append(q.values[1])

# q = belief_propagation.query(['Rain_t'], evidence={'Umbrella_t' : 1})
# kolejne_dni.append(q.values[1])

for i in range(5):
    if i == 2:
        G = FactorGraph()
        G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrella_t'])
        phi1 = DiscreteFactor(['Rain_t_1'], [2], values=np.array([q.values[0], q.values[1]]))
        phi2 = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], values=np.array([0.7, 0.3, 0.3, 0.7]))
        phi3 = DiscreteFactor(['Umbrella_t', 'Rain_t'], [2, 2], values=np.array([0.8, 0.1, 0.2, 0.9]))

        G.add_factors(phi1, phi2, phi3)

        # G.add_edges_from([('Rain_t_1', phi1), ('Rain_t', phi2), ('Rain_t_1', phi2), ('Umbrella_t', phi3), ('Rain_t', phi3)])
        G.add_edges_from([('Rain_t_1', phi1), ('Rain_t_1', phi2), ('Rain_t', phi2), ('Rain_t', phi3), ('Umbrella_t', phi3)])
        belief_propagation = BeliefPropagation(G)
        q = belief_propagation.query(['Rain_t'], evidence={'Umbrella_t': 0})

        kolejne_dni.append(q.values[1])
    else:
        G = FactorGraph()
        G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrella_t'])
        phi1 = DiscreteFactor(['Rain_t_1'], [2], values=np.array([q.values[0], q.values[1]]))
        phi2 = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], values=np.array([0.7, 0.3, 0.3, 0.7]))
        phi3 = DiscreteFactor(['Umbrella_t', 'Rain_t'], [2, 2], values=np.array([0.8, 0.1, 0.2, 0.9]))

        G.add_factors(phi1, phi2, phi3)

        # G.add_edges_from([('Rain_t_1', phi1), ('Rain_t', phi2), ('Rain_t_1', phi2), ('Umbrella_t', phi3), ('Rain_t', phi3)])
        G.add_edges_from([('Rain_t_1', phi1), ('Rain_t_1', phi2), ('Rain_t', phi2), ('Rain_t', phi3), ('Umbrella_t', phi3)])
        belief_propagation = BeliefPropagation(G)
        q = belief_propagation.query(['Rain_t'], evidence={'Umbrella_t': 1})

        kolejne_dni.append(q.values[1])

print(kolejne_dni)

