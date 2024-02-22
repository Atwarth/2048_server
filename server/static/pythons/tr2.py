def triangular(n):
    if n == 0:
        return n
    return n + triangular(n-1)
n = int(input())
print(triangular(n))

def caloriesPerServing(N,G,P,C):
    return round((9*G+4*P+4*C)/N)
caloriesPerServing(15,200,450,100)

def fibonacci(n):
    if n==0:
        return 0
    if n==1:
        return 1
    return fibonacci(n-1)+fibonacci(n-2)
n = int(input())
print(fibonacci(n))

def riman(str1,str2):
    if str1[-3:-1] == str2[-3:-1]:
        return "riman."
    return "no riman."
str1 = input()
str2 = input()
print(f"Las palabras {str1} y {str2} {riman(str1,str2)}")


while True:
    try:
        n = int(input())
        break
    except ValueError:
        print("xxx")
print("JP")