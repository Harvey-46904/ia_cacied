

 function tensor() {

  variablea=`tensor([[-0.7698,  1.3383,  0.5649,  ...,  1.3567,  0.6896, -0.1949],
    [-0.5761, -0.9789, -0.2058,  ..., -0.5843,  2.6311, -0.0008],
    [ 1.3152,  1.8851, -0.9761,  ...,  0.8639, -0.6237,  0.5646],
    ...,
    [ 0.2851,  0.5504, -0.9471,  ...,  0.0688, -0.7777,  0.1661],
    [ 2.9616, -0.8685, -1.5467,  ..., -1.4646,  1.1098, -1.0873],
    [-0.7321,  0.7610,  0.3182,  ...,  2.5859, -0.9709,  0.8537]]), dtype=torch.float64)`
  variableb=`tensor([[1.1573e+04, 6.0693e+02, 1.2436e+03, 2.7277e+04, 1.6673e+08, 2.0462e+00, 9.8891e-01],
    [2.0237e+04, 5.9074e+02, 1.7208e+03, 2.7449e+04, 2.1301e+08, 2.0678e+00, 1.0011e+00],
    [2.7456e+04, 6.1106e+02, 1.4897e+03, 2.7332e+04, 1.7310e+08, 2.0448e+00, 9.6041e-01],
    [1.7732e+04, 6.0232e+02, 1.2608e+03, 2.7371e+04, 1.8106e+08, 1.9594e+00, 1.0040e+00],
    ...,
    [1.1167e+04, 5.9867e+02, 1.3440e+03, 2.7263e+04, 2.3160e+08, 2.0190e+00, 1.0075e+00],
    [1.6003e+04, 5.9590e+02, 1.2319e+03, 2.7368e+04, 1.7155e+08, 2.0171e+00, 1.0202e+00],
    [1.5499e+04, 6.1471e+02, 9.4877e+02, 2.7395e+04, 1.8146e+08, 1.9016e+00, 9.5884e-01],
    [3.3886e+04, 6.0689e+02, 1.0777e+03, 2.7259e+04, 2.1599e+08, 2.0179e+00, 1.0201e+00]], dtype=torch.float64)`
  console.log(variablea);
}

//Recoger elemento arrastrable//
function evdragstart(ev, el) {
  //ev= el evento, el=elemento arrastrado.
  cont1 = el.parentNode; //guardamos el elemento padre del elemento en una variable.
  ev.dataTransfer.setData("text", ev.target.id); //guardamos datos del elemento.
}
//mientras se arrastra:
function evdragover(ev) {
  //ev=el evento.
  ev.preventDefault(); //quitar comportamiento por defecto.
}
//Al soltar el elemento arrastrado

let pila_divs = [];
let id = 0;

function setModal(id) {
  let letter = ""
  switch (id) {
    case "e1":
      return letter = "#m_imagen"
      break;
    case "e4":
      return letter = "#m_c45"

      break;
    case "e5":
      return letter = "#ia"
      break;
    case "e6":
      return letter = "#vista"
      break;
    case "e8":
      return letter = "#dim"
      break;
  }
}
let bloque;
let tablero;
let contador1 = 0;
let nombrese = ["Cargar Datos", "C4.5", "Red Neuronal", "Vistas"]
function evdrop(ev, el) {
  var midiv = {
    "id": id,
    "cordenada": 0
  }

  $("#msg_model").hide();
  tablero=el;
  //ev=el evento; el=receptor de soltado
  ev.stopPropagation(); //impedir otras acciones
  ev.preventDefault(); //quitar comportamiento por defecto
  var data = ev.dataTransfer.getData("text"); //recogemos datos del elemento
  mielem = ev.target.appendChild(document.getElementById(data)); //obtenemos el elemento arrastrado
  console.log(data);
  cont1.appendChild(mielem); //ponemos el elemento arrastrado en el mismo sitio donde estaba.
  let valor_modal = setModal(data)
  mielem2 = mielem.cloneNode(true); //creamos una copia del elemento arrastrado.
  mielem2.setAttribute("draggable", "false"); //impedimos que el nuevo elemento pueda volver a arrastrarse
  //titulo
  let titulo = document.createElement("h5");
  titulo.innerHTML = "EfficientNetB0"

  contador1++;
  //crear div

  let divs = document.createElement("div")
  pila_divs.push(midiv)
  divs.setAttribute("id", id);

  divs.className = " bg-cuadros text-center border border-secondary draggable"
  divs.style.cssText = "width:150px;height:180px; margin-bottom: 35px;border-radius: 20px;position: relative; z-index: 2;"
  divs.setAttribute("onmouseover", "$( function() {$( '.draggable' ).draggable() ;mirar_div(" + id + ")} );");

  divs.setAttribute("onmouseleave", "$(function() { } );");
  divs.appendChild(mielem2);
  divs.appendChild(titulo);
  //crear boton
  var button = document.createElement('button');
  button.innerHTML = 'C';
  button.className = "btn btn-primary btn-sm px-2"
  button.setAttribute("type", "button");
  button.setAttribute("data-toggle", "modal");
  button.setAttribute("data-target", valor_modal);
//crear ejecutar
  var ejectutar = document.createElement('button');
  ejectutar.innerHTML = 'E';
  ejectutar.className = "btn btn-success btn-sm px-2"
  ejectutar.setAttribute("type", "button");
  ejectutar.setAttribute("onclick", "show_image()")
//crear eliminar
var eliminar = document.createElement('button');
eliminar.innerHTML = 'D';
eliminar.className = "btn btn-danger btn-sm px-2"
eliminar.setAttribute("onclick", "eliminar(" + id + ")")
eliminar.setAttribute("type", "button");

//crear ayuda
var ayuda = document.createElement('button');
ayuda.innerHTML = 'H';
ayuda.className = "btn btn-warning btn-sm px-2 "
ayuda.setAttribute("type", "button");
  //crear boton flecha derecha
  var flecha_derecha = document.createElement('button');
  flecha_derecha.innerHTML = '.';
  flecha_derecha.className = " "
  flecha_derecha.setAttribute("type", "button");
  flecha_derecha.setAttribute("id", "r" + id);
  flecha_derecha.setAttribute("onclick", "anidar(" + id + ")")
  flecha_derecha.style.cssText = "position:absolute;bottom:60px;right:0px;"

  var flecha_izquierda = document.createElement('button');
  flecha_izquierda.innerHTML = '.';
  flecha_izquierda.className = " "
  flecha_izquierda.setAttribute("type", "button");
  flecha_izquierda.setAttribute("id", "l" + id);
  flecha_izquierda.setAttribute("onclick", "anidar(" + id + ")")
  flecha_izquierda.style.cssText = "position:absolute;bottom:60px;left:0px;"
  
  divs.appendChild(eliminar);
  divs.appendChild(ejectutar);
  /*
  divs.appendChild(button);
  divs.appendChild(flecha_derecha);
  divs.appendChild(flecha_izquierda);
  divs.appendChild(ejectutar);
  divs.appendChild(eliminar);
  divs.appendChild(ayuda);
*/
  
  //mielem2.removeAttribute("draggable")
  //mielem2.removeAttribute("id")
  mielem2.removeAttribute("ondragstart")

  el.appendChild(divs);
  
bloque=divs;
   //colocamos la copia en el receptor de soltado
  // crear_azar(4);
  id++;

}
function eliminar(id){
    var nested = document.getElementById(id);
    tablero.removeChild(nested);
    $("#msg_model").show();
    $("#msg_model_img").hide();
}
function show_image(){
  $("#msg_model_img").show();
  
}
function crear_azar(a){
  for (let index = 0; index <4; index++) {
    /*
    var p = document.getElementById("0");
      var l=p.cloneNode(true)
    tablero.appendChild(l);
    */
  
   var t=$("#0")[0]
   
   var l=t.cloneNode(true)
   l.removeAttribute("id")
   var h=l.childNodes[1];
   h.innerHTML=index
   tablero.appendChild(l);
 
  }
  
 
}