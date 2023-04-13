from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
def main():
    # Create the model - test on one set
    bayesNet = BayesianModel()

    bayesNet.add_node('Umbrella_1')
    bayesNet.add_node('Rain_0')
    bayesNet.add_node('Rain_1')

    bayesNet.add_edge('Rain_0', 'Rain_1')
    bayesNet.add_edge('Rain_1', 'Umbrella_1')

    cpd_rain0 = TabularCPD('Rain_0', 2, values=[[.4], [.6]])
    cpd_rain1 = TabularCPD('Rain_1', variable_card=2,
                                   values=[[0.7, 0.3],
                                           [0.3, 0.7]], evidence=['Rain_0'], evidence_card=[2])
    cpd_umbrella1 = TabularCPD('Umbrella_1', variable_card=2,
                                   values=[[0.8, 0.1],
                                           [0.2, 0.9]], evidence=['Rain_1'], evidence_card=[2])


    bayesNet.add_cpds(cpd_rain0, cpd_rain1, cpd_umbrella1)
    # Checking if model is correctly added
    print('Check model :', bayesNet.check_model())

    bayes_infer = VariableElimination(bayesNet)
    q = bayes_infer.query(['Rain_1'], evidence={'Umbrella_1': 1})
    print('P(Rain_1 | Umbrella_1) =\n', q)
    dzien_1 = q.values[1]
    print('dzien_1', dzien_1)


if __name__ == '__main__':
    main()