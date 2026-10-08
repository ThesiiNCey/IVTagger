# AzerothCore in AMP (Linux, Docker Compose)

Diese Vorlage macht den offiziellen AzerothCore-Docker-Stack als Generic-Instanz in AMP verfügbar. Sie installiert AzerothCore nicht als einzelne native Binärdatei: Compose startet Datenbank, Datenbankimport, Authserver, Worldserver und Clientdaten-Initialisierung gemeinsam.

## Voraussetzungen

- AMP läuft auf Linux und kann den Befehl `docker` ausführen.
- Docker Engine und das Docker-Compose-Plugin sind installiert.
- Der AMP-Dienst darf den Docker-Daemon verwenden (typischerweise Zugriff auf `/var/run/docker.sock`). Wenn AMP selbst in einem Container läuft, müssen Docker-CLI und Docker-Socket in diesen Container durchgereicht sein.
- Ausreichend freier Speicherplatz und RAM für den Quellcode, den Build, Clientdaten und die Datenbank.

Diese Vorlage ist für Docker Engine getestet gedacht. Podman kann Docker Compose teils kompatibel ausführen, ist hier aber nicht verifiziert.

## Vorlage in AMP laden

1. AMP öffnen und zu **Configuration → Instance Deployment** gehen.
2. Einen lokalen Konfigurationsordner `LOCAL-main` unter `Plugins/ADSModule/DeploymentTemplates` anlegen.
3. Die Dateien `azerothcore.kvp`, `azerothcoreconfig.json` und `azerothcoremetaconfig.json` direkt in diesen Ordner kopieren.
4. AMP Deployment-Seite neu laden und eine neue Instanz mit dem Präfix `LOCAL` und dem Namen **AzerothCore (Docker Compose)** anlegen.
5. In der neuen Instanz **Update** ausführen. AMP klont das offizielle Repo und baut die Container. Der erste Build kann lange dauern.

Der tatsächliche DeploymentTemplates-Pfad ist je nach AMP-Installation unterschiedlich. AMP dokumentiert den Standardpfad relativ zu ADS als `Plugins/ADSModule/DeploymentTemplates/LOCAL-main`.

## Datenbankpasswort vor dem ersten Start setzen

Vor dem ersten Start im AMP-Dateimanager die Datei `azerothcore-wotlk/.env` erstellen. Inhalt:

```dotenv
DOCKER_DB_ROOT_PASSWORD=HIER_EIN_LANGES_EINMALIGES_PASSWORT_EINTRAGEN
DOCKER_DB_EXTERNAL_PORT=127.0.0.1:3306
```

Ersetze den Platzhalter durch ein langes, zufälliges Passwort. Die Datenbank bleibt damit vom öffentlichen Netzwerk aus unzugänglich; AzerothCore verbindet sich intern im Compose-Netzwerk. Teile das Passwort nicht im Chat.

Danach die Instanz starten. Beim ersten Start lädt und initialisiert AzerothCore die Clientdaten und Datenbank; das kann laut offizieller Anleitung etwa 10–15 Minuten dauern. In AMP sollte die Instanz als „Running“ erscheinen, sobald Worldserver `WORLD: World Initialized` ausgibt.

## Ports

- TCP 3724: Authserver
- TCP 8085: Worldserver
- TCP 3306: Datenbank, in der Vorlage nur an `127.0.0.1` gebunden

Für Internetzugriff müssen TCP 3724 und 8085 in Firewall und Router freigegeben werden. Zusätzlich muss in der AzerothCore-Realmlist die erreichbare Adresse gesetzt werden.

## Grenzen der Vorlage

- Start und Stopp laufen über den Vordergrundprozess `docker compose up`; AMP sendet beim Stoppen SIGINT, worauf Compose die Dienste herunterfährt.
- Das AMP-Update holt den aktuellen `master`-Branch und baut die Images neu. Für reproduzierbare Builds sollte später ein fester Commit oder Tag verwendet werden.
- Dieser lokale Entwurf ist keine getestete oder offizielle CubeCoders-Vorlage. Der Quellcode-Build und die Initialisierung müssen auf dem Zielhost verifiziert werden.
