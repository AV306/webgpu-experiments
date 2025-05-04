const shaderText = `
// Depth transform matrix
const OPENGL_TO_WGPU_TRANSFORM = mat4x4f(
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0.5, 0,
  0, 0, 0.5, 1
);

@group(0) @binding(0) var<uniform> modelMatrix: mat4x4f;
@group(0) @binding(1) var<uniform> viewMatrix: mat4x4f;
@group(0) @binding(2) var<uniform> projectionMatrix: mat4x4f;

@group(1) @binding(0) var albedoSampler: sampler;
@group(1) @binding(1) var albedoTexture: texture_2d<f32>;

@group(2) @binding(0) var normalSampler: sampler;
@group(2) @binding(1) var normalTexture: texture_2d<f32>;

@group(3) @binding(0) var<uniform> lightPosition: vec3f;
//@group(3) @binding(1) var<uniform> lightDirection: vec3f;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) fragWorldPos: vec3f,
  @location(1) fragWorldNormal: vec3f,
  @location(2) fragUV: vec2f,
}

@vertex
fn vs_main(
  @location(0) position: vec3f,
  @location(1) uv: vec2f
  // @location(2) normal: vec3f
) -> VertexOutput
{
  var vertWorldPos: vec4f = modelMatrix * vec4f( position, 1 );
  
  var output: VertexOutput;
  output.position = OPENGL_TO_WGPU_TRANSFORM * projectionMatrix * viewMatrix * vertWorldPos;
  output.fragWorldPos = vertWorldPos.xyz;
  output.fragWorldNormal = vertWorldPos.xyz;
  output.fragUV = uv;
  return output;
}

@fragment
fn fs_main(
  @location(0) fragWorldPos: vec3f,
  @location(1) fragWorldNormal: vec3f,
  @location(2) fragUV: vec2f
) -> @location(0) vec4f
{
  var lightVector: vec3f = normalize( lightPosition - fragWorldPos );
  
  // Create tangent-to-world basis
  // From https://community.khronos.org/t/computing-the-tangent-space-in-the-fragment-shader/52861
  var Q1: vec3f = dpdx( fragWorldPos );
  var Q2: vec3f = dpdy( fragWorldPos );
  var st1: vec2f = dpdx( fragUV );
  var st2: vec2f = dpdy( fragUV );

  var T: vec3f = normalize( Q1*st2.y - Q2*st1.y );
  var B: vec3f = normalize( -Q1*st2.x + Q2*st1.x );

  // the transpose of texture-to-eye space matrix
  var TBN: mat3x3f = mat3x3f( T, B, fragWorldNormal );
  
  var tangentNormal: vec3f = TBN * (textureSample( normalTexture, normalSampler, fragUV ).xyz * 2 - vec3f( 1 ));
  
  // Dot light vector with surface normal
  var diffuse: f32 = dot( normalize( tangentNormal ), lightVector );
  return diffuse * textureSample( albedoTexture, albedoSampler, fragUV );
}`;


const PI = Math.PI;
const TWO_PI = Math.PI * 2;
const HALF_PI = Math.PI / 2;

const cubeVertexData = new Float32Array( [
  // Bottom face
  -0.5, -0.5, -0.5,  0, 0,
  0.5, -0.5, -0.5,   1, 0,
  -0.5, -0.5, 0.5,   0, 1,
  0.5, -0.5, 0.5,    1, 1,
  
  // Top face
  -0.5, 0.5, -0.5, 0, 0,
  0.5, 0.5, -0.5, 1, 0,
  -0.5, 0.5, 0.5, 0, 1,
  0.5, 0.5, 0.5, 1, 1,
  
  // Front face
  -0.5, -0.5, 0.5, 0, 0,
  0.5, -0.5, 0.5, 1, 0,
  -0.5, 0.5, 0.5, 0, 1,
  0.5, 0.5, 0.5, 1, 1,
  
  // Back face
  -0.5, -0.5, -0.5, 0, 0,
  0.5, -0.5, -0.5, 1, 0,
  -0.5, 0.5, -0.5, 0, 1,
  0.5, 0.5, -0.5, 1, 1,
  
  // Left Face
  -0.5, -0.5, 0.5, 0, 0,
  -0.5, -0.5, -0.5, 1, 0,
  -0.5, 0.5, 0.5, 0, 1,
  -0.5, 0.5, -0.5, 1, 1,

  // Right Face
  0.5, -0.5, 0.5, 0, 0,
  0.5, -0.5, -0.5, 1, 0,
  0.5, 0.5, 0.5, 0, 1,
  0.5, 0.5, -0.5, 1, 1
] );

const cubeIndexData = new Uint16Array( [
  0, 3, 2, 0, 1, 3,
  6, 7, 4, 7, 5, 4,
  8, 11, 10, 8, 9, 11,
  12, 14, 13, 14, 15, 13,
  16, 18, 19, 16, 19, 17,
  20, 21, 23, 20, 23, 22
] );

const vertexBufferLayout = {
  attributes: [
    {
      // Position (vec3)
      shaderLocation: 0,
      offset: 0,
      format: "float32x3"
    },
    {
      // UV (vec2)
      shaderLocation: 1,
      offset: Float32Array.BYTES_PER_ELEMENT * 3,
      format: "float32x2"
    },
    /*{
      // World-space normal (vec3)
      shaderLocation: 2,
      offset: Float32Array.BYTES_PER_ELEMENT * 5,
      format: "float32x3"
    }.*/
  ],
  //stepMode: "vertex",
  arrayStride: Float32Array.BYTES_PER_ELEMENT * 5 // 8, // 32 - vec3 (position) + vec2 (uv) + vec3 (normal)
}

const cubeVertexCount = cubeVertexData.length;
const cubeIndexDataFormat = "uint16";


const depthBufferFormat = "depth24plus";

async function setupDevice()
{
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if ( !device )
    throw Error( "browser doesn't support webgpu :(" );
  
  device.addEventListener( "uncapturederror", event => console.log( event.error.message ) );
  
  return device;
}

function setupCanvasContext( device, w=400, h=400 )
{
    const canvas = document.getElementById( "canvas" );
    canvas.width = w;
    canvas.height = h;
    const aspect = canvas.width / canvas.height;
    //canvas.style = "width: 400; height: 400;";
    
    const canvasContext = canvas.getContext( "webgpu" );
    
    canvasContext.configure( {
        device: device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "premultiplied",
    } );
    
    return [canvasContext, aspect];
}

function bindModelVertexData( device, vertexData, indexData )
{
  const vertexBuffer = device.createBuffer( {
    //size: vertexData.length * vertexBufferLayout.arrayStride,
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  } );
  
  const indexBuffer = device.createBuffer( {
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
  } );
  device.queue.writeBuffer( vertexBuffer, 0, vertexData, 0, vertexData.length );
  device.queue.writeBuffer( indexBuffer, 0, indexData, 0, indexData.length );
  
  return [vertexBuffer, indexBuffer];
}

function bindMatrixUniforms( device, pipeline, modelMatrix, viewMatrix, projectionMatrix, uniformBindGroupIndex=0 )
{
  const modelMatrixBuffer = device.createBuffer( {
    size: modelMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  const viewMatrixBuffer = device.createBuffer( {
    size: viewMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  const projectionMatrixBuffer = device.createBuffer( {
    label: "projection matrix buffer",
    size: projectionMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  device.queue.writeBuffer( modelMatrixBuffer, 0, modelMatrix );
  device.queue.writeBuffer( viewMatrixBuffer, 0, viewMatrix );
  device.queue.writeBuffer( projectionMatrixBuffer, 0, projectionMatrix );

  const uniformBindGroup = device.createBindGroup( {
    layout: pipeline.getBindGroupLayout( uniformBindGroupIndex ),
    entries: [
      {
        binding: 0,
        resource: { buffer: modelMatrixBuffer }
      },
      {
        binding: 1,
        resource: { buffer: viewMatrixBuffer }
      },
      {
        binding: 2,
        resource: { buffer: projectionMatrixBuffer }
      }
    ]
  } );
  
  return [modelMatrixBuffer, viewMatrixBuffer, projectionMatrixBuffer, uniformBindGroup];
}

function bindTextureUniforms( device, pipeline, textureData, textureSize, textureFormat, uniformBindGroupIndex )
{
  const textureBuffer = device.createTexture( {
    format: textureFormat,
    size: textureSize,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  } );
  
  console.log( textureData.BYTES_PER_ELEMENT * textureSize[0] * 4 );
  
  device.queue.writeTexture(
    { texture: textureBuffer },
    textureData,
    { bytesPerRow: textureData.BYTES_PER_ELEMENT * textureSize[0] * 4 },
    textureSize
  );
  
  //console.log( textureBuffer )
     
  const textureBindGroup = device.createBindGroup( {
    layout: pipeline.getBindGroupLayout( uniformBindGroupIndex ),
    entries: [
      {
        binding: 0,
        resource: device.createSampler()
      },
      {
        binding: 1,
        resource: textureBuffer.createView()
      }
    ]
  } );
  
  return [textureBuffer, textureBindGroup];
}

async function bindImageTextureUniforms( device, pipeline, imageUrl, textureFormat="rgba8unorm", bindGroupIndex=1 )
{
  const response = await fetch( imageUrl );
  const imageBitmap = await createImageBitmap( await response.blob() );
  
  const textureSize = [imageBitmap.width, imageBitmap.height, 1];
  
  const textureBuffer = device.createTexture( {
    size: textureSize,
    format: textureFormat,
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });
  
  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: textureBuffer },
    textureSize
  );
  
  const textureBindGroup = device.createBindGroup( {
    layout: pipeline.getBindGroupLayout( bindGroupIndex ),
    entries: [
      {
        binding: 0,
        resource: device.createSampler()
      },
      {
        binding: 1,
        resource: textureBuffer.createView()
      }
    ]
  } );
  
  return [textureBuffer, textureBindGroup];
}


function bindLightUniforms( device, pipeline, lightPos, bindGroupIndex )
{
  const uniformBuffer = device.createBuffer( {
    size: lightPos.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  device.queue.writeBuffer( uniformBuffer, 0, lightPos );
  
  const uniformBindGroup = device.createBindGroup( {
    layout: pipeline.getBindGroupLayout( bindGroupIndex ),
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer }
      }
    ]
  } );
  
  return [uniformBuffer, uniformBindGroup];
}


function createPipeline( device, presentationFormat, shaderModule )
{
  const pipeline = device.createRenderPipeline( {
    layout: "auto",
    vertex: {
      module: shaderModule,
      buffers: [ vertexBufferLayout ],
    },
    fragment: {
      module: shaderModule,
      targets: [ { format: presentationFormat } ]
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none", // Change to "back" after verifying that depth buffer works
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: depthBufferFormat,
    }
  });
  
  return pipeline;
}

function createDepthTexture( device, w=400, h=400 )
{
  const depthTexture = device.createTexture( {
    size: [w, h],
    format: depthBufferFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  } );
  return depthTexture;
}

function createRenderPassDescriptor( device, view, depthTexture, clearColor=[0, 0.5, 0.5, 1] )
{
  const renderPassDescriptor = {
    colorAttachments: [
      {
        view: view,

        clearValue: clearColor,
        loadOp: "clear",
        storeOp: "store"
      }
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store"
    }
  };
  
  return renderPassDescriptor;
}


function buildProjectionMatrix( horizontalFov, aspect, near=-1, far=-100 )
{
  // Annoyingly, all the projection matrices are for OpenGL's depth buffer which is [-1, 1], but WGPU is [0, 1]
  const xScale = 1/Math.tan( horizontalFov / 2 );
  const yScale = aspect * xScale;
  const zScaleZ = (far + near)/(near - far);
  const zScaleW = (2 * far * near)/(near - far);
  /*return new Float32Array( [
    xScale, 0, 0, 0, // Column 0 (leftmost)
    0, yScale, 0, 0,
    0, 0, zScaleZ, -1,
    0, 0, zScaleW, 0
  ] );*/
  return new Float32Array( [
    xScale, 0, 0, 0, // Column 0 (leftmost)
    0, yScale, 0, 0,
    0, 0, zScaleZ, -1,
    0, 0, -zScaleW, 0
  ] ); // Not sure why zScaleW needs to be negated, but it works, so I'm not complaining
}

// function buildProjectionMatrix( verticalFov, aspect, near=-1, far=-100 )
// {
//   /*return new Float32Array( [
//     1/(aspect * Math.tan( verticalFov / 2 )), 0, 0, 0, // Column 0 (leftmost)
//     0, 1/Math.tan( verticalFov / 2 ), 0, 0,
//     0, 0, -(far + near)/(far - near), -1,
//     0, 0, -(2 * far * near)/(far - near), 0
//   ] );*/
// }


// ========== SETUP ==========
const w = 800, h = 600;
const device = await setupDevice();
const [canvasContext, aspect] = setupCanvasContext( device, w, h );
const [modelVertexBuffer, modelIndexBuffer] = bindModelVertexData( device, cubeVertexData, cubeIndexData );

const matrixUniformBindGroupIndex = 0;
const textureUniformBindGroupIndex = 1;

// Annoyingly, matrices are COLUMN-MAJOR
let angle = 0;
let modelMatrix = new Float32Array( [
  Math.cos( angle ), 0, Math.sin( angle ), 0,
  0, 1, 0, 0,
  -Math.sin( angle ), 0, Math.cos( angle ), 0,
  0, 0, 0, 1
] );
/*const modelMatrix = new Float32Array( [
  Math.cos( angle ), -Math.sin( angle ), 0, 0,
  Math.sin( angle ), Math.cos( angle ), 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
] );*/

const cameraPitch = Math.PI/4; //PI/4;
const viewMatrix = new Float32Array( [
  1, 0, 0, 0,
  0, Math.cos( cameraPitch ), Math.sin( cameraPitch ), 0,
  0, -Math.sin( cameraPitch ), Math.cos( cameraPitch ), 0,
  0, 0, -2, 1
] );

const projectionMatrix = buildProjectionMatrix( HALF_PI, aspect );

const shaderModule = device.createShaderModule( { code: shaderText } );

const compilationInfo = await shaderModule.getCompilationInfo();
//console.log( compilationInfo )
for ( const message of compilationInfo.messages )
  console.log( message );

const pipeline = createPipeline( device, navigator.gpu.getPreferredCanvasFormat(), shaderModule );

// Bind matrix uniforms
const [modelMatrixBuffer, viewMatrixBuffer, projectionMatrixBuffer, matrixUniformBindGroup] = bindMatrixUniforms( device, pipeline, modelMatrix, viewMatrix, projectionMatrix, matrixUniformBindGroupIndex );

// Bind texture uniforms
//const [textureBuffer, textureUniformBindGroup] = bindTextureUniforms( device, pipeline, cubeTextureData, cubeTextureSize, cubeTextureFormat, textureUniformBindGroupIndex );
const [albedoTextureBuffer, albedoTextureUniformBindGroup] = await bindImageTextureUniforms( device, pipeline, "https://raw.githubusercontent.com/webgpu/webgpu-samples/main/public/assets/img/brickwall_albedo.png", "rgba8unorm", 1 );

const [normalTextureBuffer, normalTextureUniformBindGroup] = await bindImageTextureUniforms( device, pipeline, "https://raw.githubusercontent.com/webgpu/webgpu-samples/main/public/assets/img/brickwall_normal.png", "rgba8unorm", 2 );

const [lightUniformBuffer, lightUniformBindGroup] = bindLightUniforms( device, pipeline, new Float32Array( [-10, 3, 5] ), 3 );

const depthTexture = createDepthTexture( device, w, h );

const renderPassDescriptor = createRenderPassDescriptor( device, canvasContext.getCurrentTexture().createView(), depthTexture );


// ========== RENDER ==========
let prevTime = null;
function frame( time )
{
  const deltaTime = prevTime ? (time - prevTime) / 1000 : 0;
  prevTime = time;
  
  modelMatrix[0] = modelMatrix[10] = Math.cos( angle );
  modelMatrix[2] = Math.sin( angle );
  modelMatrix[8] = -Math.sin( angle );

  device.queue.writeBuffer( modelMatrixBuffer, 0, modelMatrix );
  
  renderPassDescriptor.colorAttachments[0].view = canvasContext.getCurrentTexture().createView();

  
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginRenderPass( renderPassDescriptor );
  passEncoder.setPipeline( pipeline );
  passEncoder.setBindGroup( matrixUniformBindGroupIndex, matrixUniformBindGroup );
  passEncoder.setBindGroup( 1, albedoTextureUniformBindGroup );
  passEncoder.setBindGroup( 2, normalTextureUniformBindGroup );
  passEncoder.setBindGroup( 3, lightUniformBindGroup );
  passEncoder.setVertexBuffer( 0, modelVertexBuffer );
  passEncoder.setIndexBuffer( modelIndexBuffer, cubeIndexDataFormat );
  passEncoder.drawIndexed( cubeIndexData.length );
  passEncoder.end();

  device.queue.submit( [commandEncoder.finish()] );

  angle += (Math.PI / 10) * deltaTime;
  //setTimeout( frame, 100 );
  requestAnimationFrame( frame );
  //frame();
}

frame();
