# Bilješke za odbranu

## Problem
Online prodavnica želi da poveća vrijednost korpe tako što korisniku, na osnovu trenutne korpe, preporučuje proizvode koje drugi kupci često kupuju zajedno.

## Korišćeni pristup
Rješenje kombinuje association rules, collaborative filtering signal, business margin scoring i MMR diversity reranking.

## Association rules
FP-Growth pronalazi česte kombinacije proizvoda. Iz njih se dobijaju pravila oblika:

Ako kupac uzme X i Y, vjerovatno će uzeti Z.

Za svako pravilo računaju se support, confidence i lift.

## Business vrijednost
Nije svaka statistički česta preporuka poslovno dobra. Zbog toga se u scoring dodaje margina proizvoda.

## Diversity
MMR algoritam sprječava da sistem preporuči previše slične proizvode. Cilj je da top preporuke budu i relevantne i raznovrsne.

## Demo tok
1. Izabrati proizvode u korpi.
2. Kliknuti na generisanje preporuka.
3. Pokazati preporučene proizvode.
4. Objasniti support, confidence i lift.
5. Otvoriti network graph i pokazati veze među proizvodima.

## Ograničenja
Sample dataset je mali. Na stvarnom Instacart datasetu pravila bi bila stabilnija, a collaborative filtering dio bi mogao da koristi pravi ALS model.
