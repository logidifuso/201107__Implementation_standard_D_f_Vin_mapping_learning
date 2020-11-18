#
# Implementation of learning D=f(Vin,Iout) based on
# mapping from efficiency measurement in a reference
# system
#

# The Python standard library import
import os
import shutil
# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
# Pandas for learning data
import pandas as pd
# Multiprocessing
import multiprocessing as mp

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

# The corresponding inputs (Vin, Iout) and expected outputs (D) to
# be learned
df = pd.read_csv('Lists_of_duty_cycles_values/Case_A_48Vin_12Vout_Buck/'
                 '1_Duty_cycle_values_list_csv/reduced_cases_all.csv')


def eval_fitness(net):
    """
    Evalua el fitness de los genomas de las redes neuronales
    de la población.
    Arguments:
        net: La red neuronal (feed-forward) generada por el genoma
    Returns:
        El valor del fitness (puntuación) como **el cuadrado** de la
        suma de errores absolutos tras la evaluación. Cuanto mas se
        acerque a 0, mejor.
    """
    error_sum = 0.0
    for index, row in df.iterrows():
        d = net.activate((row["VIN"], row["IOUT"]))
        error_sum += abs(d[0] - row["D"])
    # Normaliza y eleva al cuadrado para amplificar las diferencias de fitness
    error_sum /= len(df)
    fitness = (1 - error_sum)**2
    return fitness


def eval_genomes(genomas, config):
    """
    Evalúa el fitnees de cada uno de los genomas en la lista de
    genomas.
    De cada genoma, NEAT_Python genera una red neuronal feed-forward
    y la evalúa. El fitness obtenido es actualizado.
    Arguments:
        genomas:La lista de genomas de la población en la generación
                actual
        config: Los ajustes de configuración del algoritmo (valores
                de los hiperparámetros)
    """

    '''
    for genoma_id, genoma in genomas:
        # genoma.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genoma, config)
        genoma.fitness = eval_fitness(net)
    '''
    def evaluacion(genoma, config):
        genoma.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genoma, config)
        genoma.fitness = eval_fitness(net)
        return

    # Inicialización de multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # `pool.apply`
    [pool.apply(evaluacion, args=(genoma, config)) for genoma_id, genoma in genomas]

    pool.close()


def run_experiment(config_file):
    """
    Ejecuta el experimento utilizando la configuración (hiperparámetros)
    definida en el archivo de configuración.
    Genera el gráfico de red del individuo de mayour fitness y también
    muestra las estadísticas más importantes durante la ejecución del
    proceso de neuroevolución.
    Arguments:
        config_file: ruta al archivo de configuración del experimento.
    """
    # Carga configuración.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Crea la población, objeto top-level para una ejecución de NEAT
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out/neat-checkpoint-'))

    # Run for up to 300 generations.
    best_genome = p.run(eval_genomes, 300)

    # Display the best genome among generations.
    print('\nMejor genoma:\n{!s}'.format(best_genome))

    # TODO: Implementar como una función para gráficar D obtenido vs D deseado
    '''
    # Show output of the most fit genome against training data.
    print('\nSalida:')'''
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    '''
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    '''

    for index, row in df.iterrows():
        d = net.activate((row["VIN"], row["IOUT"]))
        print("input VIN={!r}, IOUT={!r}, Duty Cycle esperado {!r}, recibido {!r}"
              .format(row['VIN'], row['IOUT'], row['D'], d))







    # Comprobación de si el algoritmo tuvo éxito y notificación
    best_genome_fitness = eval_fitness(net)
    if best_genome_fitness > config.fitness_threshold:
        print("\n\nÉXITO: Duty cycles correctos!!!")
    else:
        print("\n\nFALLO: Respuesta de Duty cycles insatisfactoria!!!")

    # Visualización de los resultados del experimento
    node_names = {-1: 'Vin', -2: 'Iout', 0: 'Duty'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))


def clean_output():
    if os.path.isdir(out_dir):
        # elimina archivos creados en anteriores ejecuciones
        shutil.rmtree(out_dir)

    # creacción de la carpeta "out" con los resultados
    os.makedirs(out_dir, exist_ok=False)


if __name__ == '__main__':
    # Encuentra la ruta al archivo de configuración. Asegura la correcta
    # ejecución del script independientemente de la carpeta de trabajo
    # actual
    config_path = os.path.join(local_dir, 'mapped_duty_learning_config.ini')

    # Inicia la carpeta de salida, ..o limpia los resultados de ejecuciones
    # anteriores.
    clean_output()

    # Ejecución del experimento
    run_experiment(config_path)
