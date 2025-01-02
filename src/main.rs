mod tiles;
mod utils;

use env_logger;
use glutin::event::{Event, WindowEvent, VirtualKeyCode};
use glutin::event_loop::{ControlFlow, EventLoop};
use glutin::window::WindowBuilder;
use glutin::ContextBuilder;
use log::{error, info, LevelFilter};
use rand::thread_rng;
use rand::Rng;
use cgmath::{ortho, Matrix, Matrix4, Vector3};
use std::ffi::{CString};
use std::rc::Rc;
use std::cell::RefCell;
use std::time::{Instant, Duration};
use std::collections::HashSet;

// Bring in your tile definitions from tiles.rs
use tiles::{
    PALETTES,
    TILE_MAPPINGS,
    TILE_NAMES,
    TILE_TILE54,      // "Grass"
    TILE_TILE96,      // "Stone"
    // These eight define the "directional" Randi sprites:
    RANDI1_TILE0,     // "Randi Down Tile0 (Head)"
    RANDI1_TILE1,     // "Randi Down Tile1 (Body)"
    RANDI2_TILE0,     // "Randi Up Tile0 (Head)"
    RANDI2_TILE1,     // "Randi Up Tile1 (Body)"
    RANDI3_TILE0,     // "Randi Right Tile0 (Head)"
    RANDI3_TILE1,     // "Randi Right Tile1 (Body)"
    RANDI4_TILE0,     // "Randi Left Tile0 (Head)"
    RANDI4_TILE1,     // "Randi Left Tile1 (Body)"
};

// Bring in your utils submodules for FPS and GPU
use utils::fps::FPSCounter;
use utils::gpu::get_gpu_info;

////////////////////////////////////////////////////////////////////////////////
// 1) Direction + Part + Character
////////////////////////////////////////////////////////////////////////////////

/// The possible directions Randi can face.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Direction {
    Down,
    Up,
    Left,
    Right,
}

/// Distinguishes which part of the character sprite (head or body).
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Part {
    Head,
    Body,
}

/// Holds the current state of Randi, including position, facing direction, and textures.
struct Character {
    position: (f32, f32),
    direction: Direction,
    /// A HashMap of (Direction, Part) -> texture_id
    textures: std::collections::HashMap<(Direction, Part), u32>,
    /// We’ll track time to do “simple animation” if you want multiple frames, but for now,
    /// each direction has a unique Head/Body. You can expand frames if needed.
    last_frame_time: Instant,
    frame_duration: Duration,
    current_frame: usize, // We can keep a simple 2-frame toggle if you like
}

impl Character {
    fn new(
        initial_position: (f32, f32),
        textures: std::collections::HashMap<(Direction, Part), u32>,
        frame_duration: Duration,
    ) -> Self {
        // By default, Randi starts facing down
        Self {
            position: initial_position,
            direction: Direction::Down,
            textures,
            last_frame_time: Instant::now(),
            frame_duration,
            current_frame: 0,
        }
    }

    /// Update Randi’s state: direction, position (with delta-time), etc.
    fn update(
        &mut self,
        pressed_keys: &HashSet<VirtualKeyCode>,
        delta_time: Duration,
        window_width: f32,
        window_height: f32,
        tile_size: f32,
    ) {
        let dt = delta_time.as_secs_f32();
        let speed = 300.0; // pixels per second

        let mut moved = false;

        // Update direction & position
        if pressed_keys.contains(&VirtualKeyCode::W) || pressed_keys.contains(&VirtualKeyCode::Up) {
            self.direction = Direction::Up;
            self.position.1 += speed * dt;
            moved = true;
        }
        if pressed_keys.contains(&VirtualKeyCode::S) || pressed_keys.contains(&VirtualKeyCode::Down)
        {
            self.direction = Direction::Down;
            self.position.1 -= speed * dt;
            moved = true;
        }
        if pressed_keys.contains(&VirtualKeyCode::A) || pressed_keys.contains(&VirtualKeyCode::Left)
        {
            self.direction = Direction::Left;
            self.position.0 -= speed * dt;
            moved = true;
        }
        if pressed_keys.contains(&VirtualKeyCode::D) || pressed_keys.contains(&VirtualKeyCode::Right)
        {
            self.direction = Direction::Right;
            self.position.0 += speed * dt;
            moved = true;
        }

        // Clamp position within the window
        self.position.0 = self.position.0.clamp(0.0, window_width - tile_size);
        // Randi is 2 tiles tall (body + head)
        self.position.1 = self.position.1.clamp(0.0, window_height - 2.0 * tile_size);

        // Optional: Toggle frames if you want animation while moving
        if moved {
            let now = Instant::now();
            if now.duration_since(self.last_frame_time) >= self.frame_duration {
                // Toggle between 0 and 1
                self.current_frame = (self.current_frame + 1) % 2;
                self.last_frame_time = now;
            }
        } else {
            // If not moving, reset to frame 0
            self.current_frame = 0;
        }
    }

    /// Return the texture for the head and body based on the current direction.
    fn get_textures(&self) -> Option<(u32, u32)> {
        // Our scheme: For each direction, we have a HEAD tile and a BODY tile.
        let head_key = (self.direction, Part::Head);
        let body_key = (self.direction, Part::Body);
        let head_tex = self.textures.get(&head_key).cloned();
        let body_tex = self.textures.get(&body_key).cloned();

        match (head_tex, body_tex) {
            (Some(h), Some(b)) => Some((h, b)),
            _ => None,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// 2) Utility / Helper Functions
////////////////////////////////////////////////////////////////////////////////

/// Returns the `[R, G, B, A]` color from the palette.
#[inline]
fn palette_color(index: u8) -> [f32; 4] {
    if (index as usize) < PALETTES.len() {
        PALETTES[index as usize]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    }
}

/// Generate RGBA pixel data for a 16x16 tile.
#[inline]
fn generate_tile_pixels(tile_data: &[[u8; 16]; 16]) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(16 * 16 * 4);
    for row in tile_data {
        for &palette_index in row {
            let rgba = palette_color(palette_index);
            pixels.extend_from_slice(&rgba);
        }
    }
    pixels
}

/// Create an OpenGL texture (16x16, RGBA).
#[inline]
unsafe fn create_texture(pixels_rgba: &[f32]) -> u32 {
    let mut texture = 0;
    gl::GenTextures(1, &mut texture);
    gl::BindTexture(gl::TEXTURE_2D, texture);

    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);

    gl::TexImage2D(
        gl::TEXTURE_2D,
        0,
        gl::RGBA as i32,
        16,
        16,
        0,
        gl::RGBA,
        gl::FLOAT,
        pixels_rgba.as_ptr() as *const _,
    );
    gl::GenerateMipmap(gl::TEXTURE_2D);

    texture
}

////////////////////////////////////////////////////////////////////////////////
// 3) Shaders
////////////////////////////////////////////////////////////////////////////////

const VERT_SHADER_SRC: &str = r#"
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    uniform mat4 projection;
    uniform mat4 model;

    void main()
    {
        gl_Position = projection * model * vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
"#;

const FRAG_SHADER_SRC: &str = r#"
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;

    uniform sampler2D tileTexture;

    void main()
    {
        FragColor = texture(tileTexture, TexCoord);
        if (FragColor.a < 0.1)
            discard;
    }
"#;

/// Compile a shader from source.
unsafe fn compile_shader(src: &str, shader_type: gl::types::GLenum) -> u32 {
    let shader = gl::CreateShader(shader_type);
    let c_str = CString::new(src).unwrap();
    gl::ShaderSource(shader, 1, &c_str.as_ptr(), std::ptr::null());
    gl::CompileShader(shader);

    // Check for errors
    let mut success = gl::FALSE as gl::types::GLint;
    gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
    if success != gl::TRUE as gl::types::GLint {
        let mut len = 0;
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
        let mut buffer = Vec::with_capacity(len as usize + 1);
        buffer.extend([b' '].iter().cycle().take(len as usize));
        let error_cstring = CString::from_vec_unchecked(buffer);
        gl::GetShaderInfoLog(shader, len, std::ptr::null_mut(), error_cstring.as_ptr() as *mut _);
        error!("Shader compilation failed: {}", error_cstring.to_string_lossy());
        panic!("Shader compilation failed");
    }
    shader
}

/// Link a vertex + fragment shader into a single shader program.
unsafe fn link_program(vertex_src: &str, fragment_src: &str) -> u32 {
    let vertex_shader = compile_shader(vertex_src, gl::VERTEX_SHADER);
    let fragment_shader = compile_shader(fragment_src, gl::FRAGMENT_SHADER);

    let program = gl::CreateProgram();
    gl::AttachShader(program, vertex_shader);
    gl::AttachShader(program, fragment_shader);
    gl::LinkProgram(program);

    // Check for linking errors
    let mut success = gl::FALSE as gl::types::GLint;
    gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
    if success != gl::TRUE as gl::types::GLint {
        let mut len = 0;
        gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
        let mut buffer = Vec::with_capacity(len as usize + 1);
        buffer.extend([b' '].iter().cycle().take(len as usize));
        let error_cstring = CString::from_vec_unchecked(buffer);
        gl::GetProgramInfoLog(program, len, std::ptr::null_mut(), error_cstring.as_ptr() as *mut _);
        error!("Program linking failed: {}", error_cstring.to_string_lossy());
        panic!("Program linking failed");
    }

    gl::DeleteShader(vertex_shader);
    gl::DeleteShader(fragment_shader);

    program
}

////////////////////////////////////////////////////////////////////////////////
// 4) Tile Struct
////////////////////////////////////////////////////////////////////////////////

struct Tile {
    texture: u32,
    position: (f32, f32),
}

impl Tile {
    fn new(texture: u32, position: (f32, f32)) -> Self {
        Self { texture, position }
    }
}

////////////////////////////////////////////////////////////////////////////////
// 5) Main
////////////////////////////////////////////////////////////////////////////////

fn main() {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .init();

    // Create an event loop
    let event_loop = EventLoop::new();

    // Create window
    let window_builder = WindowBuilder::new()
        .with_title("Secret Of Mana Clone (Directional + Smooth Movement)")
        .with_inner_size(glutin::dpi::LogicalSize::new(800.0, 600.0));

    // Build GL context
    let windowed_context = ContextBuilder::new()
        .with_vsync(false)
        .build_windowed(window_builder, &event_loop)
        .expect("Failed to create windowed context");

    // Make the context current
    let windowed_context = unsafe {
        windowed_context
            .make_current()
            .expect("Failed to make context current")
    };

    // Load GL
    gl::load_with(|symbol| windowed_context.get_proc_address(symbol) as *const _);

    // Viewport + clear color
    let size = windowed_context.window().inner_size();
    unsafe {
        gl::Viewport(0, 0, size.width as i32, size.height as i32);
        gl::ClearColor(1.0, 0.0, 1.0, 1.0); // magenta
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    }

    // Create shader program
    let shader_program = unsafe { link_program(VERT_SHADER_SRC, FRAG_SHADER_SRC) };

    // Prepare background + stone textures
    let grass_pixels = generate_tile_pixels(&TILE_TILE54);
    let stone_pixels = generate_tile_pixels(&TILE_TILE96);
    let grass_texture = unsafe { create_texture(&grass_pixels) };
    let stone_texture = unsafe { create_texture(&stone_pixels) };

    // Prepare Randi’s directional textures
    let mut randi_texture_map = std::collections::HashMap::new();
    // Down
    randi_texture_map.insert((Direction::Down, Part::Head), unsafe {
        create_texture(&generate_tile_pixels(&RANDI1_TILE0))
    });
    randi_texture_map.insert((Direction::Down, Part::Body), unsafe {
        create_texture(&generate_tile_pixels(&RANDI1_TILE1))
    });
    // Up
    randi_texture_map.insert((Direction::Up, Part::Head), unsafe {
        create_texture(&generate_tile_pixels(&RANDI2_TILE0))
    });
    randi_texture_map.insert((Direction::Up, Part::Body), unsafe {
        create_texture(&generate_tile_pixels(&RANDI2_TILE1))
    });
    // Right
    randi_texture_map.insert((Direction::Right, Part::Head), unsafe {
        create_texture(&generate_tile_pixels(&RANDI3_TILE0))
    });
    randi_texture_map.insert((Direction::Right, Part::Body), unsafe {
        create_texture(&generate_tile_pixels(&RANDI3_TILE1))
    });
    // Left
    randi_texture_map.insert((Direction::Left, Part::Head), unsafe {
        create_texture(&generate_tile_pixels(&RANDI4_TILE0))
    });
    randi_texture_map.insert((Direction::Left, Part::Body), unsafe {
        create_texture(&generate_tile_pixels(&RANDI4_TILE1))
    });

    // Prepare additional (non-Randi) textures
    let mut additional_textures = Vec::new();
    for (i, &tile_data) in TILE_MAPPINGS.iter().enumerate() {
        let name = TILE_NAMES[i];
        if name.starts_with("RANDI") || name == "TILE_TILE53" {
            continue;
        }
        let pixels = generate_tile_pixels(tile_data);
        let texture = unsafe { create_texture(&pixels) };
        additional_textures.push(texture);
    }

    // Tile size
    const TILE_SIZE: f32 = 32.0;

    // Calculate how many tiles fit
    let tiles_x_init = (size.width as f32 / TILE_SIZE).ceil() as usize;
    let tiles_y_init = (size.height as f32 / TILE_SIZE).ceil() as usize;
    let tiles_x = Rc::new(RefCell::new(tiles_x_init));
    let tiles_y = Rc::new(RefCell::new(tiles_y_init));

    // Randi’s initial position (center)
    let center_x = (size.width as f32 - TILE_SIZE) / 2.0;
    let center_y = (size.height as f32 - 2.0 * TILE_SIZE) / 2.0;

    let mut character = Character::new(
        (center_x, center_y),
        randi_texture_map,
        Duration::from_millis(200), // Frame toggle time (for your animation if desired)
    );

    // Additional random tiles
    let mut rng = thread_rng();
    let mut extra_tiles = Vec::new();
    for &texture in &additional_textures {
        let x = rng.gen_range(0.0..(size.width as f32 - TILE_SIZE));
        let y = rng.gen_range(0.0..(size.height as f32 - TILE_SIZE));
        extra_tiles.push(Tile::new(texture, (x, y)));
    }
    let extra_tiles_ref = Rc::new(RefCell::new(extra_tiles));

    // Random stones
    let mut stones_vec = Vec::new();
    for _ in 0..5 {
        let x = rng.gen_range(0.0..(size.width as f32 - TILE_SIZE));
        let y = rng.gen_range(0.0..(size.height as f32 - TILE_SIZE));
        stones_vec.push(Tile::new(stone_texture, (x, y)));
    }
    let stones_ref = Rc::new(RefCell::new(stones_vec));

    // Pressed keys
    let pressed_keys = Rc::new(RefCell::new(HashSet::new()));

    // GPU + FPS
    unsafe {
        let gpu_info = get_gpu_info();
        info!("GPU Vendor: {}", gpu_info.vendor);
        info!("GPU Renderer: {}", gpu_info.renderer);
        info!("OpenGL Version: {}", gpu_info.version);
    }
    let mut fps_counter = FPSCounter::new();

    // Create VAO + VBO
    let (mut vao, mut vbo) = (0, 0);
    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);

        gl::GenBuffers(1, &mut vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

        // Two triangles for a tile
        let vertices: [f32; 24] = [
            // x, y,   u, v
            0.0, 0.0, 0.0, 1.0,
            TILE_SIZE, 0.0, 1.0, 1.0,
            TILE_SIZE, TILE_SIZE, 1.0, 0.0,

            0.0, 0.0, 0.0, 1.0,
            TILE_SIZE, TILE_SIZE, 1.0, 0.0,
            0.0, TILE_SIZE, 0.0, 0.0,
        ];
        let size_bytes = (vertices.len() * std::mem::size_of::<f32>()) as isize;
        gl::BufferData(gl::ARRAY_BUFFER, size_bytes, vertices.as_ptr() as *const _, gl::STATIC_DRAW);

        // position attribute
        gl::VertexAttribPointer(
            0,
            2,
            gl::FLOAT,
            gl::FALSE,
            (4 * std::mem::size_of::<f32>()) as i32,
            std::ptr::null(),
        );
        gl::EnableVertexAttribArray(0);

        // texcoord attribute
        gl::VertexAttribPointer(
            1,
            2,
            gl::FLOAT,
            gl::FALSE,
            (4 * std::mem::size_of::<f32>()) as i32,
            (2 * std::mem::size_of::<f32>()) as *const _,
        );
        gl::EnableVertexAttribArray(1);
    }

    // Log which tiles we loaded
    info!(
        "Tiles loaded (excluding Randi + tile53): {:?}",
        TILE_NAMES
            .iter()
            .filter(|&&n| !n.starts_with("RANDI") && n != "TILE_TILE53")
            .copied()
            .collect::<Vec<&str>>()
    );

    info!("OpenGL program linked. Starting main loop...");

    // Orthographic projection
    let projection = ortho(0.0, size.width as f32, 0.0, size.height as f32, -1.0, 1.0);
    let proj_cstr = CString::new("projection").unwrap();
    let proj_loc = unsafe {
        gl::UseProgram(shader_program);
        gl::GetUniformLocation(shader_program, proj_cstr.as_ptr())
    };
    unsafe {
        gl::UseProgram(shader_program);
        gl::UniformMatrix4fv(proj_loc, 1, gl::FALSE, projection.as_ptr());
    }

    // Track time for smooth movement
    let mut last_frame_time = Instant::now();

    // Run event loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => {
                // Request redraw
                windowed_context.window().request_redraw();
            }

            Event::RedrawRequested(_) => {
                // Calculate delta_time
                let now = Instant::now();
                let delta_time = now.duration_since(last_frame_time);
                last_frame_time = now;

                // Clear + set program
                unsafe {
                    gl::Clear(gl::COLOR_BUFFER_BIT);
                    gl::UseProgram(shader_program);
                    gl::BindVertexArray(vao);
                }

                // Draw background grass
                let current_tiles_x = *tiles_x.borrow();
                let current_tiles_y = *tiles_y.borrow();

                unsafe {
                    gl::BindTexture(gl::TEXTURE_2D, grass_texture);
                }
                for ty in 0..current_tiles_y {
                    for tx in 0..current_tiles_x {
                        let pos_x = tx as f32 * TILE_SIZE;
                        let pos_y = ty as f32 * TILE_SIZE;
                        let model = Matrix4::from_translation(Vector3::new(pos_x, pos_y, 0.0));
                        let model_loc = unsafe {
                            gl::GetUniformLocation(shader_program, CString::new("model").unwrap().as_ptr())
                        };
                        unsafe {
                            gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.as_ptr());
                            gl::DrawArrays(gl::TRIANGLES, 0, 6);
                        }
                    }
                }

                // Draw stones
                for stone in stones_ref.borrow().iter() {
                    unsafe {
                        gl::BindTexture(gl::TEXTURE_2D, stone.texture);
                    }
                    let model =
                        Matrix4::from_translation(Vector3::new(stone.position.0, stone.position.1, 0.0));
                    let model_loc = unsafe {
                        gl::GetUniformLocation(shader_program, CString::new("model").unwrap().as_ptr())
                    };
                    unsafe {
                        gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.as_ptr());
                        gl::DrawArrays(gl::TRIANGLES, 0, 6);
                    }
                }

                // Draw additional tiles
                for tile in extra_tiles_ref.borrow().iter() {
                    unsafe {
                        gl::BindTexture(gl::TEXTURE_2D, tile.texture);
                    }
                    let model =
                        Matrix4::from_translation(Vector3::new(tile.position.0, tile.position.1, 0.0));
                    let model_loc = unsafe {
                        gl::GetUniformLocation(shader_program, CString::new("model").unwrap().as_ptr())
                    };
                    unsafe {
                        gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.as_ptr());
                        gl::DrawArrays(gl::TRIANGLES, 0, 6);
                    }
                }

                // Update Randi (smooth movement + direction)
                character.update(
                    &pressed_keys.borrow(),
                    delta_time,
                    size.width as f32,
                    size.height as f32,
                    TILE_SIZE,
                );

                // Draw Randi’s head and body
                if let Some((head_tex, body_tex)) = character.get_textures() {
                    // Body
                    unsafe {
                        gl::BindTexture(gl::TEXTURE_2D, body_tex);
                    }
                    let model_body = Matrix4::from_translation(Vector3::new(character.position.0, character.position.1, 0.0));
                    let model_loc_body = unsafe {
                        gl::GetUniformLocation(shader_program, CString::new("model").unwrap().as_ptr())
                    };
                    unsafe {
                        gl::UniformMatrix4fv(model_loc_body, 1, gl::FALSE, model_body.as_ptr());
                        gl::DrawArrays(gl::TRIANGLES, 0, 6);
                    }

                    // Head: one tile above
                    unsafe {
                        gl::BindTexture(gl::TEXTURE_2D, head_tex);
                    }
                    let model_head = Matrix4::from_translation(Vector3::new(
                        character.position.0,
                        character.position.1 + TILE_SIZE,
                        0.0,
                    ));
                    unsafe {
                        gl::UniformMatrix4fv(model_loc_body, 1, gl::FALSE, model_head.as_ptr());
                        gl::DrawArrays(gl::TRIANGLES, 0, 6);
                    }
                }

                // Swap buffers
                windowed_context.swap_buffers().unwrap();

                // Update FPS
                if let Some(fps) = fps_counter.update() {
                    info!("FPS: {}", fps);
                }
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    info!("Close requested, exiting.");
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(new_size) => {
                    unsafe {
                        gl::Viewport(0, 0, new_size.width as i32, new_size.height as i32);
                    }
                    let new_projection = ortho(
                        0.0,
                        new_size.width as f32,
                        0.0,
                        new_size.height as f32,
                        -1.0,
                        1.0,
                    );
                    unsafe {
                        gl::UseProgram(shader_program);
                        gl::UniformMatrix4fv(
                            gl::GetUniformLocation(shader_program, proj_cstr.as_ptr()),
                            1,
                            gl::FALSE,
                            new_projection.as_ptr(),
                        );
                    }

                    // Update tile counts
                    let new_tiles_x = (new_size.width as f32 / TILE_SIZE).ceil() as usize;
                    let new_tiles_y = (new_size.height as f32 / TILE_SIZE).ceil() as usize;
                    *tiles_x.borrow_mut() = new_tiles_x;
                    *tiles_y.borrow_mut() = new_tiles_y;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(keycode) = input.virtual_keycode {
                        match input.state {
                            glutin::event::ElementState::Pressed => {
                                pressed_keys.borrow_mut().insert(keycode);
                            }
                            glutin::event::ElementState::Released => {
                                pressed_keys.borrow_mut().remove(&keycode);
                            }
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    });
}
