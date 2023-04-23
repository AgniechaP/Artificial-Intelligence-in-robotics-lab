# P(Rt+1|u1:t+1) = P(Rt+1 | U1:t,U1:t+1) = alpha*P(Ut+1|Rt+1,U1:t)P(Rt+1|U1:t)=
# = alpha*P(Ut+1|Rt+1)P(Rt+1|U1:t) = alpha*P(Ut+1|Rt+1)*Suma(
# alpha*P(Ut+1|Rt+1) - model sensora
# P(Rt+1|Rt) - model ruchu
# P(Rt|U1:t) - to co w poprzednim stanie

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

bayesNet = BayesianModel()

bayesNet.add_node('Umbrella_t')
bayesNet.add_node('Rain_t')
bayesNet.add_node('Rain_t_1')

bayesNet.add_edge('Rain_t_1', 'Rain_t')
bayesNet.add_edge('Rain_t', 'Umbrella_t')

cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, values=[[.4], [.6]])
cpd_rain_t = TabularCPD('Rain_t', variable_card=2,
                       values=[[0.7, 0.3],
                               [0.3, 0.7]], evidence=['Rain_t_1'], evidence_card=[2])
cpd_umbrella_t = TabularCPD('Umbrella_t', variable_card=2,
                                   values=[[0.8, 0.1],
                                           [0.2, 0.9]], evidence=['Rain_t'], evidence_card=[2])

bayesNet.add_cpds(cpd_rain_t_1, cpd_rain_t, cpd_umbrella_t)
# Checking if model is correctly added
print('Check model :', bayesNet.check_model())
print(bayesNet.number_of_edges())

bayes_infer = VariableElimination(bayesNet)

# 5.1 Za pomocą pgmpy zamodeluj rozkład dla sekwencji 5 obserwacji. Jaki jest rozkład stanu dla chwili 5 mając dany ciąg obserwacji true, true, false, true, true?

kolejne_dni = []
for i in range(5):
    if i == 2:
        q = bayes_infer.query(['Rain_t'], evidence={'Umbrella_t': 0})
        kolejne_dni.append(q.values[1])
    else:
        q = bayes_infer.query(['Rain_t'], evidence={'Umbrella_t': 1})
        kolejne_dni.append(q.values[1])
    cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
    bayesNet.add_cpds(cpd_rain_t_1)
print('P(R1 | U1) =\n', q)

# 5.2
print(kolejne_dni)

# 5.3 Predykcja - policz rozkłady dla chwili 6 i 9
q = bayes_infer.query(['Rain_t'])
print('Dla dnia 6 =\n', q)
cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
bayesNet.add_cpds(cpd_rain_t_1)
q = bayes_infer.query(['Rain_t'])
cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
bayesNet.add_cpds(cpd_rain_t_1)
q = bayes_infer.query(['Rain_t'])
cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
bayesNet.add_cpds(cpd_rain_t_1)
q = bayes_infer.query(['Rain_t'])
print('Dla dnia 9 =\n', q)


