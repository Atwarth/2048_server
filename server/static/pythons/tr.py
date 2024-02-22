#!/usr/bin/env python
# coding: utf-8

# In[89]:


def sumatoria(n,p):
    if n==0:
        return n
    return n*p+sumatoria(n-1,p)
n = int(input())
p = int(input())
print(sumatoria(n,p))


# In[ ]:





# In[90]:


import datetime

dia = int(input())
mes = int(input())
anyo = int(input())
date = datetime.date(anyo,mes,dia)
christmas = datetime.date(anyo,12,25)
print("Current Date: {} {}, {}.".format(date.strftime("%B"),date.strftime("%d"),date.strftime("%Y")))
print("There are {} days left for Christmas.".format((christmas - date).days))
print("This year, Christmas is on {}.".format(christmas.strftime("%A")))


# In[ ]:





# In[70]:





# In[88]:


def descifrar(frase):    
    ap = 'abcdefghijklmnopqrstuvwxyz'
    ap = sorted(set(ap))
    dic = {}
    for i,letra in enumerate(ap):
        if i<25:
            dic[letra] = ap[i+1]
        else:
            dic[letra] = ap[0]
            dic[" "] = " "
    frase_des = ""
    for i in frase:
        frase_des+= dic[i]
    return frase_des
n = int(input())
lista_frases = []
for i in range(n):
    lista_frases.append(input())
for f in lista_frases:
    print(descifrar(f))


# In[ ]:





# In[87]:


def calcularValorPorPersona(V,P,N):
    total = V+V*P/100
    return round(total/N,1)

V = float(input())
P = float(input())
N = float(input())
print(f"Cada persona debe poner {calcularValorPorPersona(V,P,N)} pesos.")


# In[ ]:




