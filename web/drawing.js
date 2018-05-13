/* © 2009 ROBO Design
 * http://www.robodesign.ro
 */

// Keep everything in anonymous function, called on window load.
if(window.addEventListener) {
window.addEventListener('load', function () {
	var canvas, context, tool;

	function init () {
		// Find the canvas element.
		canvas = document.getElementById('imageView');
		if (!canvas) {
			alert('Error: Cannot find the canvas element!');
			return;
		}

		canvas.width = canvas.clientWidth;
		canvas.height = canvas.clientHeight;

		if (!canvas.getContext) {
			alert('Error: no canvas.getContext!');
			return;
		}

		// Get the 2D canvas context.
		context = canvas.getContext('2d');
		if (!context) {
			alert('Error: failed to getContext!');
			return;
		}

		context.beginPath();
		context.rect(0, 0, 280, 280);
		context.fillStyle = "black";
		context.fill();

		// Pencil tool instance.
		tool = new tool_pencil();

		// Attach the mousedown, mousemove and mouseup event listeners.
		canvas.addEventListener('mousedown', ev_canvas, false);
		canvas.addEventListener('touchstart', ev_canvas, false);
		canvas.addEventListener('mousemove', ev_canvas, false);
		canvas.addEventListener('touchmove', ev_canvas, false);
		canvas.addEventListener('mouseup',	 ev_canvas, false);
		canvas.addEventListener('touchend', ev_canvas, false);
	}

	// This painting tool works like a drawing pencil which tracks the mouse 
	// movements.
	function tool_pencil () {
		var tool = this;
		this.started = false;

		// This is called when you start holding down the mouse button.
		// This starts the pencil drawing.
		this.mousedown = function (ev) {
				context.beginPath();
				context.lineWidth=20;
				context.miterLimit = 4;
				context.lineJoin = "round";
				context.lineCap="round";
				context.moveTo(ev.offsetX, ev.offsetY);
				tool.started = true;
				tool.lastX = ev.offsetX;
				tool.lastY = ev.offsetY;
		};

		this.touchstart = function (ev) {
			this.mousedown(ev);
			event.stopPropagation();
			event.preventDefault();
		};

		// This function is called every time you move the mouse. Obviously, it only 
		// draws if the tool.started state is set to true (when you are holding down 
		// the mouse button).
		this.mousemove = function (ev) {
			if (tool.started) {
                //if( Math.sqrt( Math.abs( ev.offsetX - tool.lastX ) + Math.abs( ev.offsetY - tool.lastY ) ) > 2 )
                {
					context.lineTo(ev.offsetX, ev.offsetY);
					context.strokeStyle = "#ff0000";
					context.stroke();
					
					tool.lastX = ev.offsetX;
					tool.lastY = ev.offsetY;
				}
			}
		};

		this.touchmove = function (ev) {
			this.mousemove(ev);
			event.stopPropagation();
			event.preventDefault();
		};

		// This is called when you release the mouse button.
		this.mouseup = function (ev) {
			if (tool.started) {
				tool.mousemove(ev);
				tool.started = false;
			}
		};

		this.touchend = function (ev) {
			this.mouseup(ev);
			event.stopPropagation();
			event.preventDefault();
		};
	}

	// The general-purpose event handler. This function just determines the mouse 
	// position relative to the canvas element.
	function ev_canvas (ev) {
		if (ev.layerX || ev.layerX == 0) { // Firefox
			ev._x = ev.layerX;
			ev._y = ev.layerY;
		} else if (ev.offsetX || ev.offsetX == 0) { // Opera
			ev._x = ev.offsetX;
			ev._y = ev.offsetY;
		}

		// Call the event handler of the tool.
		var func = tool[ev.type];
		if (func) {
			func(ev);
		}
	}

	init();

}, false); }
