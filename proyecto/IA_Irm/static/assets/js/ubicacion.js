function ubicacion(a,r){
    let id_formt = "#" + a;
    console.log(id_formt)
    var coordenadas = $(id_formt).position();
    console.log("Y: " + coordenadas.top + " X: " + coordenadas.left)
    console.log(JSON.stringify(pila_divs))
    document.getElementById("informacion").innerHTML = JSON.stringify(pila_divs);
}
let relacion=0;
let padre=0;
let madre=0;
let relaciones = [];
function anidar(a){
    switch (relacion) {
        case 0:
            relacion++;
            console.log( mousex+" "+ mousey);
            padre=a;
            var estandar = document.createElementNS('http://www.w3.org/2000/svg','line');
            estandar.setAttribute('id',"estandar1");
            estandar.setAttribute('x1',15);
            estandar.setAttribute('y1',91);
            estandar.setAttribute('x2',15);
            estandar.setAttribute('y2',91); 
            estandar.style.cssText ="  stroke:rgb(0,0,0);stroke-width:3;"
                $("#map").append(estandar);
           
            
           
           
        break;
        case 1:
            madre=a;
            var pareja = {
                "desde": padre,
                "hacia": madre
              }
            relaciones.push(pareja);
            relacion=0;
            document.getElementById("relaciones").innerHTML = JSON.stringify(relaciones);
           const [x,y,x1,y1]= cordenadas(padre,madre);
            crear_linea(x,y,x1,y1);
        break;
    
    }
}
let line=0;
function crear_linea(x,y,x1,y1){
    var newLine = document.createElementNS('http://www.w3.org/2000/svg','line');
    newLine.setAttribute('id','line'+line);
    newLine.setAttribute('x1',y);
    newLine.setAttribute('y1',x);
    newLine.setAttribute('x2',y1);
    newLine.setAttribute('y2',x1); 
    newLine.style.cssText ="  stroke:rgb(0,0,0);stroke-width:3;"
    $("#map").append(newLine);
  
    line++;
}
function cordenadas(padre,madre){
   var respadre = pila_divs.find(conector => conector.id === padre)
  
  const [x,y]= interceptar_cordenada(respadre.cordenada)
   var resmadre = pila_divs.find(conector => conector.id === madre)
 
  const [x1,y1]= interceptar_cordenada(resmadre.cordenada)
  return [x,y,x1,y1];
}
function interceptar_cordenada(nodo){
var palabra=nodo.replace(/ /g, "$")
var dividir=palabra.split("$")
return [
   dividir[1],
   dividir[3]
]

}
