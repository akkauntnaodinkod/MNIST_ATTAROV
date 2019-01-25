import numpy
import scipy.special
import matplotlib.pyplot 

# определение класса нейронной сети 
class neuralNetwork:
    
    
    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество узлов во входном, скрытом и выходном слое 
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Матрицы весовых коэффициентов связей wih (между входным и скрытым слоями)и who (между скрытым и выходным слоями). 
        # Весовые коэффициенты связей между узлом i и узлом j следующего слоя обозначены как w__i_j : 
        # w11 w21
        # w12 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # коэффициент обучения 
        self.lr = learningrate
        
        # использование сигмоиды в качестве функции активации 
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # тренировка нейронной сети 
    def train(self, inputs_list, targets_list):
        #  преобразование списка входных значений в двухмерный массив 
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # рассчитать входящие сигналы для скрытого слоя 
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя 
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # рассчитать входящие сигналы для выходного слоя 
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя 
        final_outputs = self.activation_function(final_inputs)
        
        #  ошибки выходного слоя = (целевое значение - фактическое значение) 
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors, распределенные пропорционально весовым коэффициентам связей и рекомбинированные на скрытых узлах 
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        #  обновить веса для связей между скрытым и выходным слоями 
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # обновить весовые коэффициенты для связей между входным и скрытым слоями 
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # рассчитать входящие сигналы для скрытого слоя 
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя 
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # рассчитать входящие сигналы для выходного слоя 
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя 
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
# количество входных, скрытых и выходных узлов 
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэффициент обучения 
learning_rate = 0.1

## создать экземпляр нейронной сети 
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

#загрузить в список тестовый набор данных CSV-файла набора MNIST

training_data_file = open("mnist_train.csv", 'r') 
training_data_list = training_data_file.readlines() 
training_data_file.close()

# тренировка нейронной сети

# переменная epochs указывает, сколько раз тренировочный набор данных используется для тренировки сети

epochs = 5
for е in range(epochs):

    # перебрать все записи в тренировочном наборе данных 
    for record in training_data_list:
        #  получить список значений из записи, используя символы запятой (',') в качестве разделителей 
        all_values = record.split(',')
        # масштабировать и сместить входные значения 
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создать целевые выходные значения (все равны 0,01, за сключением желаемого маркерного значения, равного 0,99) 
        targets = numpy.zeros(output_nodes) + 0.01
    
        #all_values[0] целевое маркерное значение # для данной записи 
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# загрузить в список тестовый набор данных CSV-файла набора MNIST 
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines() 
test_data_file.close()

#получить первую тестовую запись
all_values = test_data_list[0].split(',')
# вывести маркер
print(all_values[0])

image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
"""print(n.query((numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01))"""

# тестирование нейронной сети

# журнал оценок работы сети, первоначально пустой 
scorecard = []
for record in test_data_list:
    # получить список значений из записи, используя символы 
    # запятой (',') в качестве разделителей 
    all_values = record.split(',')
    # правильный ответ - первое значение 
    correct_label = int(all_values[0])
    """print(correct_label, "истинный маркер") """
    # масштабировать и сместить входные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 
    # опрос сети 
    outputs = n.query(inputs)
    # индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs) 
    """ print(label, "ответ сети")  """
    # присоединить оценку ответа сети к концу списка 
    if (label == correct_label) : 
        # в случае правильного ответа сети присоединить к списку значение 1
        scorecard.append(1) 
    else:
        # в случае неправильного ответа сети присоединить к списку значение 0
        scorecard.append(0) 
        pass

    pass

"""print(scorecard)"""

# рассчитать показатель эффективности в виде доли правильных ответов
scorecard_array = numpy.asarray(scorecard)
print ("эффективность = ", scorecard_array.sum() / scorecard_array.size)
































































