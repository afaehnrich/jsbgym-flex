# Installation, Beschreibung und Nutzung von JSBSim als Reinforcement Learning Umgebung

Rene Titze und Andreas Fähnrich


 
## 1	Einleitung
JSBSim berechnet die Flugdynamik von Luftfahrzeugen als physisch, mathematisches 6 DoF-Modell.
1.  Luftfahrzeuge
2.  Environment (Atmosphäre, Wind, Gravity, etc.)
3.  other
Mittels  JSBSim  und  gymjsbsim  kann  der  flugdynamische  Prozess  als  Markov  Decision  Prozess  (MDP) abgebildet werden.
```
import gym
import gymjsbsim
env = gym.make(ENVID)
env.reset()
state, reward, done, info = env.step(action)
```
## 2	Installation
### 2.1	JSBSim
Die Simulation JSBSim ist ein Bestandteil des Flugsimulationsprogramms Flightgear, kann jedoch auch losgelöst von diesem als flugdynamisches Modell genutzt werden. JSBSim ist weitestgehend in C++ geschrieben, es gibt jedoch ein Python-Interface, welches als Modul installiert werden kann:
pip install jsbsim
### 2.2	JSBSim Flugzeugmodelle
Die zu JSBSim gehörigen Flugzeug- und Antriebsmodelle werden nicht mit dem Python-Modul zusammen installiert. Diese können von der Github-Seite des JSBSim-Projektes heruntergeladen werden: https://github.com/JSBSim-Team/jsbsim Ordner "Aircraft" und "Engine".
Es können auch eigene Flugzeugmodelle definiert werden. Näheres dazu findet sich im JSBSim Reference Manual

### 2.3	Gym_jsbsim
Installation ins Python-Env: 
```
pip install git+https://github.com/Gor-Ren/gym-jsbsim
```
### 2.4	Jsbgym-flex
Jsbgym-flex ist ein fork von gym-jsbsim mit einigen Anpassungen.
Installation ins Python-Env:
```
pip install git+https://github.com/afaehnrich/jsbgym-flex
```
Anpassungen:
* Konfiguration weitestgehend über eine Konfigurationsdatei im TOML-Format
* Multi-Agent Learning

## 3	Anpassen des Environments jsbgym-flex

Das Environment jsbgym-flex kann mittels einer Konfigurationsdatei im TOML-Format angepasst werden.
Die Konfigurationsdatei wird beim Aufruf des Konstruktors übergeben.
```
env = jsbgym_flex.environment.JsbSimEnv(cfg = cfgfile, shaping = Shaping.STANDARD)
```
Das Environment wird hierbei nicht mit der Funktion env.make(ENVID) intitialisiert, da dabei keine individuelle Konfiguration des Environments möglich wäre. Das standardisierte OpenAI-Gym zielt darauf ab, verschiedene Reinforcement Learnign Algorithmen an standardisierten RL-Umgebungen zu testen. Jsbgym-flex zielt darauf ab, eine möglichst flexible Lernumgebung zu schaffen und diese zu optimieren.
In der Konfigurationsdatei können folgende Einstellungen vorgenommen werden:

### 3.1	Abschnitt „environment“
Im Abschnitt Environment wird die RL-Umgebung konfiguriert.
Es können folgende Parameter gesetzt werden:

Parameter|Typ|Erklärung
---------|---|------------
simulation_stepwidth|real|Aktualisierungsintervall der Simulation (JSBSim) in Sekunden
controller_stepwidth|real|Aktualisierungsintervall der PID- und Servocontroller in Sekunden
observation_stepwidth|real|Beobachtungsintervall der RL-Umgebung in Sekunden. So oft werden Beobachtungen an den RL-Agent geliefert.
aircraft = "cessna172P"|String|Name des zu simulierenden Flugzeugs. Siehe aircraft.py (TODO: aircraft.py entfernen?)
path_jsbsim|String|Pfad, in dem sich die Verzeichnisse „aircraft“ und „engine“ mit den JSBSim-Flugzeugs- und Antriebsmodellen befinden
Pids	Array of|String|Array mit aktiven pids (siehe Abschnitt „pid“)

