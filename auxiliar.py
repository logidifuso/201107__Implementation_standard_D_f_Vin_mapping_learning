
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def reducir_csv(filename='Lists_of_duty_cycles_values/'
                         'Case_A_48Vin_12Vout_Buck/1_Duty_cycle_values_list_csv/'
                         '201106_Duty_estimation_from_measurements_for_learning.csv'):
    """
    Creacción de un archivo .csv reducido con sólo los valores cada 10 ó 20 filas
    :param filename: ruta al archvo .csv original
    :type filename: str
    :return:
    """
    # The corresponding inputs (Vin, Iout) and expected outputs (D) to
    # be learned
    df = pd.read_csv(filename)

    # IMPORTANTE: Recuerda utilizar el operador "*" para desempaquetar (argument-unpacking)
    indices = [*range(10, len(df), 20)]

    # Opcionalmente también podemos utilizar la función extend() para desempaquetar el
    # resultado del range
    '''
    indices = []

    # Valores para el rango
    start, end, interval = 10, len(df), 20

    # Comprobación de que el valor inicial no es menor que el final
    if start < end:
        # desempaqueta el resultado
        indices.extend(range(start, end, interval))
        # Anexa el último valor
        indices.append(end)
    '''

    # Slicing del dataframe df and saving it to new .csv file
    df2 = df.iloc[indices]
    reduced_cases_all_Vin = df2.to_csv('Lists_of_duty_cycles_values/'
                                       'Case_A_48Vin_12Vout_Buck/1_Duty_cycle_values_list_csv/'
                                       'reduced_cases_all.csv', index=False)
    return


def graf_contorno1(df_result, out_dir):
    """
    Graficado de las salidas obtenidas por la mejor red neuronal evolucionada
    Parameters
    ----------
    df_result : pd.DataFrame
        Dataframe que contiene las salidas producidas por la red neuronal
    out_dir : str
        Dirección donde guarda el gráfico generado
    Returns
    -------
    """
    result = df_result.pivot_table(index='x', columns='y', values='z_result').T.values
    target = df_result.pivot_table(index='x', columns='y', values='z_train').T.values
    x_unicos = np.sort(df_result.x.unique())
    y_unicos = np.sort(df_result.y.unique())
    x, y = np.meshgrid(x_unicos, y_unicos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.contourf(x, y, target)
    ax1.set_title('Función objetivo')

    im2 = ax2.contourf(x, y, result)
    ax2.set_title('Resultado obtenido')

    cbar = fig.colorbar(im2, ax=ax2)

    plt.savefig(os.path.join(out_dir, 'predicciones.svg'))
    plt.show()
    plt.close()
    return


# *********************************************************************************************
#       Testeo
# *********************************************************************************************
'''
local_dir = os.path.dirname(__file__)
out_dir = os.path.join(local_dir, 'out_2_variab_funct')
df = pd.read_csv('out_2_variab_funct/resultados.csv')
graf_contorno1(df, out_dir)
'''

