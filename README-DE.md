# AzerothCore in AMP – lokale native Linux-Vorlagen

Dieses Paket fügt zwei AMP-Generic-Vorlagen hinzu: **AzerothCore Auth Server** und **AzerothCore World Server**. AMP kann damit die bereits installierten AzerothCore-Prozesse starten, stoppen und ihre Konsolen anzeigen.

## Voraussetzung

AzerothCore muss bereits nativ auf dem Linux-Host kompiliert und eingerichtet sein. Die Standardvorlage erwartet:

- `/opt/azerothcore/env/dist/bin/authserver`
- `/opt/azerothcore/env/dist/bin/worldserver`

Die Vorlagen laden oder kompilieren AzerothCore nicht und richten auch Datenbank oder Clientdaten nicht ein. Im AMP-Konfigurationsfeld **AzerothCore Installationspfad** lässt sich `/opt/azerothcore` ändern.

**Bei AMP-Instanzen in Docker/Podman:** Der Host-Ordner mit AzerothCore muss zusätzlich als Mount in den Container eingebunden sein. Binde den Host-Pfad (standardmäßig `/opt/azerothcore`) im Container unter demselben Pfad ein. Ohne diesen Mount kann AMP die Vorlage zwar laden, der Serverprozess findet die Binärdateien beim Start aber nicht. Der Mount muss für den AMP-Container lesbar sein; falls AzerothCore dort Logs oder Konfigurationen schreibt, muss er auch schreibbar sein.

## Installation der Vorlagen

1. Lade dieses ZIP auf den AMP-Host.
2. Entpacke es in den ADS-Deployment-Templates-Ordner, typischerweise:
   `/home/amp/.ampdata/instances/ADS01/Plugins/ADSModule/DeploymentTemplates/`
3. Der ZIP-Ordner `LOCAL-main` muss direkt dort liegen. Darin müssen die beiden `.kvp`-Dateien, vier JSON-Dateien und `manifest.json` direkt liegen.
4. Stelle sicher, dass der AMP-Benutzer die Dateien lesen kann. Falls du als root hochgeladen hast, kannst du ausführen:
   `chown -R amp:amp /home/amp/.ampdata/instances/ADS01/Plugins/ADSModule/DeploymentTemplates/LOCAL-main`
5. Lade die AMP-Seite neu. Falls die Vorlagen noch nicht auftauchen, starte nur das ADS-Panel neu:
   `su - amp -c 'ampinstmgr restart ADS01'`
6. Erstelle zwei neue Instanzen: **AzerothCore Auth Server** und **AzerothCore World Server**. Starte zuerst Auth Server, danach World Server.

## Ports

- Authserver: TCP 3724
- Worldserver: TCP 8085

Öffne diese Ports in der Host-Firewall und im Router, wenn Spieler von außerhalb deines Netzwerks verbinden sollen. Richte außerdem die AzerothCore-Realm-Adresse in der Datenbank passend ein.

## Konsolensteuerung

Der Worldserver bietet eine beschreibbare Konsole in AMP. Der Authserver wird ebenfalls überwacht und kann in AMP gestartet und gestoppt werden. Beide Prozesse verwenden SIGTERM zum Herunterfahren.
