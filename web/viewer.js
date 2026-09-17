(function () {
  const canvas = document.getElementById("c");
  const gl = canvas.getContext("webgl");
  const info = document.getElementById("info");
  if (!gl) {
    info.textContent = "当前浏览器不支持 WebGL";
    return;
  }

  function resize() {
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener("resize", resize);
  resize();

  const vs = `
    attribute vec3 aPos;
    attribute vec3 aCol;
    uniform mat4 uMVP;
    uniform float uPoint;
    varying vec3 vCol;
    void main() {
      gl_Position = uMVP * vec4(aPos, 1.0);
      gl_PointSize = uPoint;
      vCol = aCol;
    }
  `;
  const fs = `
    precision mediump float;
    varying vec3 vCol;
    void main() {
      vec2 p = gl_PointCoord * 2.0 - 1.0;
      float d = dot(p, p);
      if (d > 1.0) discard;
      float a = exp(-d * 2.2);
      gl_FragColor = vec4(vCol, a);
    }
  `;

  function compile(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    return sh;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.DEPTH_TEST);

  const locPos = gl.getAttribLocation(prog, "aPos");
  const locCol = gl.getAttribLocation(prog, "aCol");
  const locMVP = gl.getUniformLocation(prog, "uMVP");
  const locPoint = gl.getUniformLocation(prog, "uPoint");

  let N = 0;
  let center = [0, 0, 0];
  let radius = 2;
  let yaw = 0.6, pitch = 0.4, dist = 4;
  let panX = 0, panY = 0;
  let dragging = false, panning = false, lastX = 0, lastY = 0;

  canvas.addEventListener("mousedown", (e) => {
    if (e.button === 2) panning = true;
    else dragging = true;
    lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { dragging = panning = false; });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  window.addEventListener("mousemove", (e) => {
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragging) { yaw += dx * 0.01; pitch += dy * 0.01; pitch = Math.max(-1.4, Math.min(1.4, pitch)); }
    if (panning) { panX += dx * 0.002 * dist; panY -= dy * 0.002 * dist; }
  });
  canvas.addEventListener("wheel", (e) => {
    dist *= e.deltaY > 0 ? 1.08 : 0.92;
    e.preventDefault();
  }, { passive: false });
  document.getElementById("fs").onclick = () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
  };
  const resetBtn = document.getElementById("reset");
  if (resetBtn) {
    resetBtn.onclick = () => {
      yaw = 0.6; pitch = 0.4; panX = 0; panY = 0; dist = radius * 2.4;
    };
  }

  function matMul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
      o[i*4+j] = a[i*4]*b[j] + a[i*4+1]*b[4+j] + a[i*4+2]*b[8+j] + a[i*4+3]*b[12+j];
    return o;
  }
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const m = new Float32Array(16);
    m[0] = f / aspect; m[5] = f; m[10] = (far + near) / (near - far); m[11] = -1;
    m[14] = (2 * far * near) / (near - far);
    return m;
  }
  function lookAt(eye, target, up) {
    function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
    function norm(a){const l=Math.hypot(a[0],a[1],a[2])||1; return [a[0]/l,a[1]/l,a[2]/l];}
    function cross(a,b){return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];}
    const z = norm(sub(eye, target));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    const m = new Float32Array(16);
    m[0]=x[0]; m[4]=x[1]; m[8]=x[2]; m[12]=-(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);
    m[1]=y[0]; m[5]=y[1]; m[9]=y[2]; m[13]=-(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);
    m[2]=z[0]; m[6]=z[1]; m[10]=z[2]; m[14]=-(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);
    m[15]=1;
    return m;
  }

  function parsePly(buf) {
    const text = new TextDecoder("ascii").decode(buf.slice(0, Math.min(buf.byteLength, 8192)));
    const headerEnd = text.indexOf("end_header");
    if (headerEnd < 0) throw new Error("不是有效 PLY");
    const header = text.slice(0, headerEnd);
    const vertexMatch = header.match(/element vertex\s+(\d+)/);
    N = vertexMatch ? parseInt(vertexMatch[1], 10) : 0;
    const props = [];
    header.split(/\r?\n/).forEach((line) => {
      const m = line.match(/property\s+(\w+)\s+(\w+)/);
      if (m) props.push({ type: m[1], name: m[2] });
    });
    const little = /format binary_little_endian/.test(header);
    const ascii = /format ascii/.test(header);
    const headerBytes = buf.slice(0, headerEnd).byteLength + ("end_header\n").length;
    // handle CRLF
    let offset = new TextDecoder("ascii").decode(buf.slice(0, headerEnd + 20)).indexOf("end_header");
    offset += "end_header".length;
    if (new Uint8Array(buf)[offset] === 13) offset += 1;
    if (new Uint8Array(buf)[offset] === 10) offset += 1;

    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    let cx = 0, cy = 0, cz = 0;
    const getIdx = (name) => props.findIndex((p) => p.name === name);
    const ix = getIdx("x"), iy = getIdx("y"), iz = getIdx("z");
    const dc0 = getIdx("f_dc_0"), dc1 = getIdx("f_dc_1"), dc2 = getIdx("f_dc_2");
    const red = getIdx("red"), green = getIdx("green"), blue = getIdx("blue");

    if (ascii) {
      const body = new TextDecoder().decode(buf.slice(offset));
      const lines = body.trim().split(/\r?\n/);
      for (let i = 0; i < N && i < lines.length; i++) {
        const p = lines[i].trim().split(/\s+/).map(Number);
        pos[i*3] = p[ix]||0; pos[i*3+1]=p[iy]||0; pos[i*3+2]=p[iz]||0;
        if (dc0>=0) {
          col[i*3] = 0.5 + (p[dc0]||0) * 0.28209479;
          col[i*3+1] = 0.5 + (p[dc1]||0) * 0.28209479;
          col[i*3+2] = 0.5 + (p[dc2]||0) * 0.28209479;
        } else if (red>=0) {
          col[i*3]= (p[red]||0)/255; col[i*3+1]=(p[green]||0)/255; col[i*3+2]=(p[blue]||0)/255;
        } else { col[i*3]=col[i*3+1]=col[i*3+2]=0.8; }
        cx += pos[i*3]; cy += pos[i*3+1]; cz += pos[i*3+2];
      }
    } else {
      const view = new DataView(buf);
      const stride = props.reduce((s, p) => s + (p.type === "uchar" || p.type === "uint8" ? 1 : 4), 0);
      for (let i = 0; i < N; i++) {
        const base = offset + i * stride;
        let off = 0;
        const values = [];
        for (const p of props) {
          if (p.type === "uchar" || p.type === "uint8") {
            values.push(view.getUint8(base + off));
            off += 1;
          } else {
            values.push(little ? view.getFloat32(base + off, true) : view.getFloat32(base + off, false));
            off += 4;
          }
        }
        pos[i*3] = values[ix]||0; pos[i*3+1]=values[iy]||0; pos[i*3+2]=values[iz]||0;
        if (dc0>=0) {
          col[i*3] = 0.5 + (values[dc0]||0) * 0.28209479;
          col[i*3+1] = 0.5 + (values[dc1]||0) * 0.28209479;
          col[i*3+2] = 0.5 + (values[dc2]||0) * 0.28209479;
        } else if (red>=0) {
          col[i*3]=(values[red]||0)/255; col[i*3+1]=(values[green]||0)/255; col[i*3+2]=(values[blue]||0)/255;
        } else { col[i*3]=col[i*3+1]=col[i*3+2]=0.8; }
        cx += pos[i*3]; cy += pos[i*3+1]; cz += pos[i*3+2];
      }
    }
    center = [cx / Math.max(N,1), cy / Math.max(N,1), cz / Math.max(N,1)];
    let maxd = 0.01;
    for (let i = 0; i < N; i++) {
      const dx = pos[i*3]-center[0], dy=pos[i*3+1]-center[1], dz=pos[i*3+2]-center[2];
      maxd = Math.max(maxd, Math.hypot(dx,dy,dz));
    }
    radius = maxd;
    dist = radius * 2.4;
    const pb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pb);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locPos);
    gl.vertexAttribPointer(locPos, 3, gl.FLOAT, false, 0, 0);
    const cb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cb);
    gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locCol);
    gl.vertexAttribPointer(locCol, 3, gl.FLOAT, false, 0, 0);
    info.textContent = `Gaussian / 点：${N}`;
  }

  function frame() {
    const eye = [
      center[0] + dist * Math.cos(pitch) * Math.cos(yaw) + panX,
      center[1] + dist * Math.sin(pitch) + panY,
      center[2] + dist * Math.cos(pitch) * Math.sin(yaw),
    ];
    const proj = perspective(60 * Math.PI/180, canvas.width / Math.max(canvas.height, 1), 0.01, radius * 40);
    const view = lookAt(eye, [center[0]+panX, center[1]+panY, center[2]], [0,1,0]);
    gl.uniformMatrix4fv(locMVP, false, matMul(proj, view));
    gl.uniform1f(locPoint, Math.max(2, 180 / dist));
    gl.clearColor(0.16, 0.14, 0.12, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (N) gl.drawArrays(gl.POINTS, 0, N);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  const params = new URLSearchParams(location.search);
  const src = params.get("src") || "/api/missing.ply";
  fetch(src).then((r) => {
    if (!r.ok) throw new Error("无法加载 PLY");
    return r.arrayBuffer();
  }).then(parsePly).catch((err) => { info.textContent = String(err); });
})();
