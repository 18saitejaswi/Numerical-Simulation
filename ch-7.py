'''class People():
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def greet(self):
        print("Greetings, " + self.name)
person1 = People(name = 'Iron Man', age = 35)
person1.greet()
print(person1.name)
print(person1.age)'''

'''class People():
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def greet(self):
        print("Greetings, " + self.name)
person2 = People(name = 'Batman', age = 33)
person2.greet()
print(person2.name)
print(person2.age)'''


'''class Student():
    
    n_instances = 0
    
    def __init__(self, sid, name, gender):
        self.sid = sid
        self.name = name
        self.gender = gender
        self.type = 'learning'
        Student.n_instances += 1
        
    def say_name(self):
        print("My name is " + self.name)
        
    def report(self, score):
        self.say_name()
        print("My id is: " + self.sid)
        print("My score is: " + str(score))
        
    def num_instances(self):
        print(f'We have {Student.n_instances}-instance in total')'''
        
        
'''class Sensor():
    def __init__(self, name, location, record_date):
        self.name = name
        self.location = location
        self.record_date = record_date
        self.data = {}
        
    def add_data(self, t, data):
        self.data['time'] = t
        self.data['data'] = data
        print(f'We have {len(data)} points saved')        
        
    def clear_data(self):
        self.data = {}
        print('Data cleared!')'''
        
        
        
'''class Sensor():
    def __init__(self, name, location):
        self.name = name
        self._location = location
        self.__version = '1.0'
    
    # a getter function
    def get_version(self):
        print(f'The sensor version is {self.__version}')
    
    # a setter function
    def set_version(self, version):
        self.__version = version
sensor1 = Sensor('Acc', 'Berkeley')
print(sensor1.name)
print(sensor1._location)
print(sensor1.__version)'''
