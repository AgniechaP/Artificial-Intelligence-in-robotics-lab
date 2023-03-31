# Zadanie 4, code inspiration from http://anmolkapoor.in/2019/05/05/Inference-Bayesian-Networks-Using-Pgmpy-With-Social-Moderator-Example/

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def main():
    # Create the model
    bayesNet = BayesianModel()

    bayesNet.add_node('Battery')
    bayesNet.add_node('Radio')
    bayesNet.add_node('Ignition')
    bayesNet.add_node('Gas')
    bayesNet.add_node('Starts')
    bayesNet.add_node('Moves')

    bayesNet.add_edge('Battery', 'Radio')
    bayesNet.add_edge('Battery', 'Ignition')
    bayesNet.add_edge('Ignition', 'Starts')
    bayesNet.add_edge('Gas', 'Starts')
    bayesNet.add_edge('Starts', 'Moves')

    # Adding CPDs for each node. A quick note is that while adding proabilities, we have to give FALSE values first.
    cpd_bat = TabularCPD('Battery', 2, values=[[.3], [.7]])
    cpd_gas = TabularCPD('Gas', 2, values=[[.5], [.5]])
    cpd_igni = TabularCPD(variable='Ignition', variable_card=2,
                          values=[[1.00, 0.03],
                                  [0.0, 0.97]], evidence=['Battery'], evidence_card=[2])
    cpd_rad = TabularCPD(variable='Radio', variable_card=2,
                         values=[[1.0, 0.1],
                                 [0.0, 0.9]], evidence=['Battery'], evidence_card=[2])
    cpd_mov = TabularCPD(variable='Moves', variable_card=2,
                         values=[[1.00, 0.2],
                                 [0.00, 0.8]], evidence=['Starts'], evidence_card=[2])
    cpd_mov_starts = TabularCPD(variable='Starts', variable_card=2,
                                values=[[1.00, 1.00, 1.00, 0.05],
                                        [0.00, 0.00, 0.00, 0.95]], evidence=['Ignition', 'Gas'], evidence_card=[2, 2])
    bayesNet.add_cpds(cpd_bat, cpd_gas, cpd_igni,cpd_rad, cpd_mov, cpd_mov_starts)

    # Checking if model is correctly added
    print('Check model :', bayesNet.check_model())

    # Checking independiences
    # print('Independencies:\n', bayesNet.get_independencies())

    # Initialize inference algorithm
    bayes_infer = VariableElimination(bayesNet)

    # Tutaj (w porownaniu do zadania 3. - 1 to true, 0 to false, bo w TabularCPD najpierw ida wartosci false)
    # 2. P(Starts | Gas, Radio)
    q = bayes_infer.query(['Starts'], evidence={'Radio': 1, 'Gas': 1})
    print('P(Starts | Gas, Radio) =\n', q)
    # Result Starts(1) = 0,9215 = Starts(True)

    # 3. P(Battery | Moves)
    q = bayes_infer.query(['Battery'], evidence={'Moves' : 1})
    print(q)
    # Result Battery(1) = 1
if __name__ == '__main__':
    main()