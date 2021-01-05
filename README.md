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

#### 3.1.1	Unterabschnitt „initial_state“    
In diesem Abschnitt können Eigenschaften (Properties) der Flugsimulation initial gesetzt werden. Zu den Eigenschaften siehe Abschnitt „Properties“ und JSBSim Reference Manual.   
Beispiel:
```
    [environment.initial_state]
        initial_altitude_ft = 5000
```
#### 3.1.2	Unterabschnitt „init_sequence“
In diesem Abschnitt können weitere Eigenschaften der Flugsimulation gesetzt werden, welche erst nach dem Initial state gesetzt werden sollten
Beispiel:
```    
    [environment.init_sequence]
        all_engine_running = -1 # -1 = running; 0 = off
        throttle_cmd = 0.8
        mixture_cmd = 0.8
```
### 3.2	Abschnitt „tasks“
In diesem Abschnitt werden die Reinforcement Learning-Tasks definiert. Es können mehrere Tasks definiert werden (multi-agent learning). Für jeden Task gibt es einen eigenen Unterabschnitt.
Beispiel:
```
[tasks]
    [tasks.FlyAlongLine]
  actor = 'lin_4x128'
    	  critic = 'lin_4x128'
        task_name = 'FlyAlongLine'
        actions = ['target_heading']
        observations = ['roll_rad', 'heading_rad']
        [tasks.FlyAlongLine.init]
            head_target = 0.2
```
Parameter|Typ|Erklärung
---------|---|---------
actor|String|Typ des Actor-Netzwerks (bei Actor-Critic Verfahren). Wird vom Modul RL-wrapper ausgewertet.
critic|String|Typ des Critic-Netzwerks (bei Actor-Critic Verfahren). Wird vom Modul RL-wrapper ausgewertet.
task_name|String|Typ des Tasks. Siehe jsbgym-flex -> tasks.py -> task_dict
Actions []|Array of Property|Array mit den Eigenschaften (Properties), welche jsbgym-flex als Actions erwartet
Observations []|Array of Property|Array mit den Eigenschaften (Properties), welche jsbgym-flex als observation/state zurückliefert

#### 3.2.1	Unterabschnitt „init“
In diesem Abschnitt können Task-spezifische Initialiserungen vorgenommen werden. Diese müssen im Task ausgelesen werden.
### 3.3	Abschnitt „pid“
In diesem Abschnitt können PID-Regler konfiguriert werden. Für jeden PID-Regler gibt es einen eigenen Unterabschnitt. Der Name des Unterabschnitts ist auch der Name des PID-Reglers
Beispiel:
```
[pid]

    [pid.heading]
        type = 'pid_angle'
        input = 'heading_rad'
        output = 'target_roll'
        target = 'target_heading'
        angle_max = 6.28318530718 # 2 * math.pi
        out_min = -0.75
        out_max = 0.75
        p = 0.7
        i = 0.0 
        d = 0.0
        anti_windup = 2
```
Parameter|String|Erklärung
---------|------|---------
type|String|Typ des PID-Reglers. “pid_angle” meint einen PID-Regler, welcher für Winkel nutzbar ist (mit Überlauf bei 360°)
input|Property|Regelgröße/Istwert des Reglers (JSBSim-Property)
output|Property|Stellgröße des Reglers (JSBSim-Property)
target|Property|Führungsgröße/Sollwert des Reglers (JSBSim-Property)
out_min|real|Untere Begrenzung der Stellgröße
out_max|real|Obere Begrenzung der Stellgröße
p|real|Verstärkung des Proportionalreglers
i|real|Verstärkung des Integralreglers
d|real|Verstärkung des Differentialreglers
anti_windup|real|Bei Vorzeichenwechsel der Regelabweichung wird wird das Integral durch diesen Wert geteilt, um eine Sättigung des Integrals zu verhindern
angle_max|real|(Bei Typ „pid_angle“) Größter möglicher Winkel. Zur korrekten Berücksichtigung des Überlaufs.

### 3.4	Abschnitt „visualiser“
In diesem Abschnitt wird die graphische Darstellung der Simulation konfiguriert.
Parameter|Typ|Erklärung
---------|---|---------
enable|Boolean|Graphische Darstellung an/aus

Es können verschiedene Arten der Visualiserung definiert werden, welche jeweils in einem Unterabschnitt definiert sind. Derzeit gibt es nur die Visualisierungstypen „flightgear“ und „figure“. Für Typ „figure“ gibt es keine weiteren Parameter.
#### 3.4.1	Unterabschnitt „flightgear“
Parameter|Typ|Erklärung
---------|---|---------
timefactor|Integer|Faktor, um den die Darstellung der Simulation im Verhältnis zur Realzeit beschleunigt wird. Größer = schneller
start_fgfs|Boolean|Gibt an, ob fgfs aus Python heraus gestartet wird oder ob der Benutzer es vorher selbst starten muss. Ein Start durch den Benutzer führte zu weniger Instabilitäten.
Startparameter für flightgear bei start_fgfs=true:
time|String|Tageszeit der Simulation
type|String|Verbindungstyp
direction|String|Verbindungsrichtung (‘in’ oder ‘out’)
rate|String|Wiederholungsrate der Flugsimulation in Herz
server|String|Servername/-adresse
port|String|Server-Port
protocol|String|Server-Protokoll (sollte „UDP“ sein)

Beispiel:
```
[visualiser]

    [visualiser.flightgear]
        timefactor = 
        start_fgfs = false
        time = 'noon'
        type = 'socket'
        direction = 'in'
        rate = 60
        server = ''
        port = 5550
        protocol = 'udp'
```
### 3.5	Abschnitt „properties“
In diesem Abschnitt werden Eigenschaften der Flugsimulation definiert, welche in jsbsim-flex benutzt werden können. Es kann auf vorhandene Eigenschaften lesend und schreibend zugegriffen werden (siehe JSBSim Reference Manual) oder es können eigene Eigenschaften definiert werden.
Alle Eigenschaften sind vom Datentyp real. Es können obere und untere Grenzen für den Wertebereich definiert werden (Bound Property) oder es gilt der gesamte Wertebereich (Unbound Property).
Für jede Eigenschaft gibt es einen eigenen Unterabschnitt.
Beispiel:
```
[properties]
#custom properties
    [properties.target_roll]
        name = 'custom/target-roll'
        description = 'target for PID roll'
        min = -0.7 #-1.57079632679 #-0.5 * math.pi
        max = 0.7  #1.57079632679 #0.5 * math.pi
# position and attitude
    [properties.altitude_sl_ft]
        name = 'position/h-sl-ft'
        description = 'altitude above mean sea level [ft]'
        min = -1400
        max = 85000
# initial conditions
    [properties.initial_altitude_ft]
        name = 'ic/h-sl-ft'
        description = 'initial altitude MSL [ft]'
```
Parameter|Typ|Erklärung
---------|---|---------
name|String|Name des Properties in JSBSim. Für vorhandene Eigenschaften vgl. JSBSim Reference Manual
description|String|Beschreibung der Eigenschaft. Wird derzeit nicht genutzt.
Für Bound Properties:
min|Real|Untere Grenze des Wertebereichs
max|Real|Obere Grenze des Wertebereichs

## 4	Docker
### 4.1	Jsbsim im Docker

