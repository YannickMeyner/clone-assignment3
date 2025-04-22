# Implementieren Sie mehrere Versionen einer GPU beschleunigten Matrix-Multiplikation

Folgende Vorgaben gelten:
-	Sie dürfen für ihre eignen Implementationen keine fremden Libraries zur Hilfe nehmen.
-	Verwenden Sie die Struktur und Helper-Funktionen der bereits eigeführten CUDA-Beispiele.
-	Verwenden Sie folgendes Projekt als Grundlage: https://classroom.github.com/a/GW7FTIwX
-	Sie können die Code Vorlage beliebig Ändern, sofern der Kern der Berechnung noch gleichbleibt. 

## Bewertung

Implementierungen
-	½P für eine CUDA basierte Matrix-Multiplikation, welche hauptsächlich das Global-Memory benutzt
-	½P für eine cuBLAS basierte Matrix-Multiplikation
-	1½P für eine CUDA basierte Version, welche das Shared-Memory korrekt benutzt und so tendenziell schneller ist als vorherige Version.
-	2P für eine CUDA basierte Version, welche tendenziell noch schneller/optimierter ist als die vor-herige Shared-Memory Version.
-	½P für eine optimierte CPU Version (hier darf auch eine Library verwendet werden). 

Dokumentation:
-	½P für die sinnvolle Untersuchungen/Messungen mit den CUDA Profiling Tools und deren Analyse. Ein sauberes Zeit-Profiling aller gewählten Ansätze (ebenso vs CPU Code). Unterscheiden Sie zwischen Init-Aufwand und der wiederholten Ausführung des Codes/der Kernel.
-	½P für eine saubere Dokumentation der verwendeten Ansätze und des allgemeinen Aufbaus des Codes. Sowie sinnvollen Schlussfolgerungen der erarbeiteten Resultate. Saubere Gegenüberstel-lung der Runtime aller implementierten Versionen über verschiedene Matrix-Grössen.

Falls keine Dokumentation zum Code abgegeben wird, gehe ich davon aus, dass der Code abge-schrieben / mit ChatGPT erstellt wurde. In diesem Fall werden keine Punkte vergeben.

Die Abgabe erfolgt über einen Pull Request. Erstellen Sie am besten zu Beginn einen «dev» Branch, vom welchem Sie dann in den «main» einen Pull Request erstellen. Laden sie den Dozenten (Simon Marcin) als Reviewer in den Pull Request ein.  
