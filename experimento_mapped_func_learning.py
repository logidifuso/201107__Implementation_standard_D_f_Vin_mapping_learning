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

import pandas as pd
import numpy as np

# Fuciones auxiliares para mis experimentos
import visualize
# The helper used to visualize experiment results
import auxiliar as aux

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out_2_variab_funct')

# The corresponding inputs (Vin, Iout) and expected outputs (D) to
# be learned
df = pd.read_csv('Lists_of_duty_cycles_values/funciones/'
                 '201111_sin_x2_2y2.csv')


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
        d = net.activate((row["x"], row["y"]))
        error_sum += abs(d[0] - row["z"])
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
    for genoma_id, genoma in genomas:
        # genoma.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genoma, config)
        genoma.fitness = eval_fitness(net)


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

    # Annade un informe en "stdout" y muestra el progreso en la terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out_2_variab_funct/neat-checkpoint-'))

    # todo : Incrementa el número de generaciones (ahora está a 10 a pesar del comentario)
    # Ejecuta hasta 300 generaciones
    mejor_genoma = p.run(eval_genomes, 2)

    # Muestra el mejor genoma (de todas las generaciones)
    print('\nMejor genoma:\n{!s}'.format(mejor_genoma))

    net = neat.nn.FeedForwardNetwork.create(mejor_genoma, config)

    # TODO: Implementar como una función para gráficar D obtenido vs D deseado
    # Creacción del dataframe con los resultados y el conjunto de entrenamiento
    zz_results = np.zeros(len(df.index))
    for index, row in df.iterrows():
        z_result = net.activate((row["x"], row["y"]))
        zz_results[index] = z_result[0]  # OJO: z_result es una lista de 1 solo elemento
        # Opcional -- Muestra las salidas obtenidas con el mejor genoma comparadas con la respuesta
        # correcta

        print("input x={!r}, y={!r}, z esperado {!r}, recibido {!r}"
              .format(row['x'], row['y'], row['z'], z_result))

    df_result = pd.DataFrame()
    df_result['x'] = df['x']
    df_result['y'] = df['y']
    df_result['z_train'] = df['z']
    df_result['z_result'] = zz_results
    # Conversión y grabado en archivo 'csv'
    file_extension = ".csv"
    df_result.to_csv(os.path.join(out_dir, "resultados" + file_extension), index=False)

    # Crea y guarda gráfica de contorno con los resultados obtenidos
    aux.graf_contorno(df_result, out_dir)

    # Comprobación de si el algoritmo tuvo éxito y notificación
    best_genome_fitness = eval_fitness(net)
    if best_genome_fitness > config.fitness_threshold:
        print("\n\nÉXITO: Se aprendió la función en el intervalo especificado!!!")
    else:
        print("\n\nFALLO: No se aprendió la función satisfactoriamente!!!")

    # Visualización de los resultados del experimento
    node_names = {-1: 'x', -2: 'y', 0: 'z'}
    visualize.draw_net(config, mejor_genoma, True, node_names=node_names, directory=out_dir)
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
