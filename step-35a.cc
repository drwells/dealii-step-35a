/* ---------------------------------------------------------------------
 * Copyright (C) 2009 - 2017 by the deal.II authors
 *
 * This file is NOT part of the deal.II library.
 *
 * This file is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Abner Salgado, Texas A&M University 2009
 * Author: David Wells, Virginia Tech, 2014
 *         David Wells, Rensselaer Polytechnic Institute, 2017
 */


#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <boost/archive/text_oarchive.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#define STEP_35A_USE_PETSC

#ifdef STEP_35A_USE_PETSC
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#endif

#include "h5.h"

namespace Step35
{

  using namespace dealii;

#ifdef STEP_35A_USE_PETSC
  inline
  void
  copy_to_petsc(const Vector<double> &input,
                PETScWrappers::MPI::Vector &result)
  {
    AssertDimension(input.size(), result.size());
    PetscScalar *solution_ptr;

    int ierr = VecGetArray(result, &solution_ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    std::copy(input.begin(), input.end(), solution_ptr);
    ierr = VecRestoreArray(result, &solution_ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }

  inline
  void
  copy_from_petsc(Vector<double> &result,
                  PETScWrappers::MPI::Vector &input)
  {
    AssertDimension(input.size(), result.size());
    PetscScalar *solution_ptr;

    int ierr = VecGetArray(input, &solution_ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    std::copy(solution_ptr, solution_ptr + input.size(), result.begin());
    ierr = VecRestoreArray(input, &solution_ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }
#endif

  namespace RunTimeParameters
  {
    enum MethodFormulation
    {
      METHOD_STANDARD,
      METHOD_ROTATIONAL
    };

    class Data_Storage
    {
    public:
      Data_Storage();
      ~Data_Storage();
      void read_data (const char *filename);
      MethodFormulation form;
      double initial_time,
             final_time,
             Reynolds;
      std::string load_checkpoint_file_name;
      std::string save_checkpoint_file_name;
      double dt;
      unsigned int n_global_refines,
               pressure_degree;
      unsigned int vel_max_iterations,
               vel_Krylov_size,
               vel_off_diagonals,
               vel_update_prec;
      double vel_eps,
             vel_diag_strength;
      bool verbose;
      unsigned int output_interval;
    protected:
      ParameterHandler prm;
    };


    Data_Storage::Data_Storage()
    {
      prm.declare_entry ("Method_Form", "rotational",
                         Patterns::Selection ("rotational|standard"),
                         " Used to select the type of method that we are going "
                         "to use. ");
      prm.enter_subsection ("Physical data");
      {
        prm.declare_entry ("initial_time", "0.",
                           Patterns::Double (0.),
                           " The initial time of the simulation. ");
        prm.declare_entry ("final_time", "1.",
                           Patterns::Double (0.),
                           " The final time of the simulation. ");
        prm.declare_entry ("Reynolds", "1.",
                           Patterns::Double (0.),
                           " The Reynolds number. ");
        prm.declare_entry ("load_checkpoint_file_name", "", Patterns::Anything(),
                           "file containing the checkpoint data.");
        prm.declare_entry ("save_checkpoint_file_name", "", Patterns::Anything(),
                           "file in which to save checkpoint data.");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Time step data");
      {
        prm.declare_entry ("dt", "5e-4",
                           Patterns::Double (0.),
                           " The time step size. ");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Space discretization");
      {
        prm.declare_entry ("n_of_refines", "0",
                           Patterns::Integer (0, 15),
                           " The number of global refines we do on the mesh. ");
        prm.declare_entry ("pressure_fe_degree", "1",
                           Patterns::Integer (1, 5),
                           " The polynomial degree for the pressure space. ");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Data solve velocity");
      {
        prm.declare_entry ("max_iterations", "1000",
                           Patterns::Integer (1, 1000),
                           " The maximal number of iterations GMRES must make. ");
        prm.declare_entry ("eps", "1e-12",
                           Patterns::Double (0.),
                           " The stopping criterion. ");
        prm.declare_entry ("Krylov_size", "30",
                           Patterns::Integer(1),
                           " The size of the Krylov subspace to be used. ");
        prm.declare_entry ("off_diagonals", "60",
                           Patterns::Integer(0),
                           " The number of off-diagonal elements ILU must "
                           "compute. ");
        prm.declare_entry ("diag_strength", "0.01",
                           Patterns::Double (0.),
                           " Diagonal strengthening coefficient. ");
        prm.declare_entry ("update_prec", "15",
                           Patterns::Integer(1),
                           " This number indicates how often we need to "
                           "update the preconditioner");
      }
      prm.leave_subsection();

      prm.declare_entry ("verbose", "true",
                         Patterns::Bool(),
                         " This indicates whether the output of the solution "
                         "process should be verbose. ");

      prm.declare_entry ("output_interval", "1",
                         Patterns::Integer(1),
                         " This indicates between how many time steps we print "
                         "the solution. ");
    }



    Data_Storage::~Data_Storage()
    {}



    void Data_Storage::read_data (const char *filename)
    {
      std::ifstream file (filename);
      AssertThrow (file, ExcFileNotOpen (filename));

      prm.parse_input (file);

      if (prm.get ("Method_Form") == std::string ("rotational"))
        form = METHOD_ROTATIONAL;
      else
        form = METHOD_STANDARD;

      prm.enter_subsection ("Physical data");
      {
        initial_time = prm.get_double ("initial_time");
        final_time   = prm.get_double ("final_time");
        Reynolds     = prm.get_double ("Reynolds");
        load_checkpoint_file_name = prm.get ("load_checkpoint_file_name");
        save_checkpoint_file_name = prm.get ("save_checkpoint_file_name");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Time step data");
      {
        dt = prm.get_double ("dt");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Space discretization");
      {
        n_global_refines = prm.get_integer ("n_of_refines");
        pressure_degree     = prm.get_integer ("pressure_fe_degree");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Data solve velocity");
      {
        vel_max_iterations = prm.get_integer ("max_iterations");
        vel_eps            = prm.get_double ("eps");
        vel_Krylov_size    = prm.get_integer ("Krylov_size");
        vel_off_diagonals  = prm.get_integer ("off_diagonals");
        vel_diag_strength  = prm.get_double ("diag_strength");
        vel_update_prec    = prm.get_integer ("update_prec");
      }
      prm.leave_subsection();

      verbose = prm.get_bool ("verbose");

      output_interval = prm.get_integer ("output_interval");
    }
  }


  namespace EquationData
  {
    template <int dim>
    class MultiComponentFunction: public Function<dim>
    {
    public:
      MultiComponentFunction (const double initial_time = 0.);
      void set_component (const unsigned int d);
    protected:
      unsigned int comp;
    };

    template <int dim>
    MultiComponentFunction<dim>::
    MultiComponentFunction (const double initial_time)
      :
      Function<dim> (1, initial_time), comp(0)
    {}


    template <int dim>
    void MultiComponentFunction<dim>::set_component(const unsigned int d)
    {
      Assert (d<dim, ExcIndexRange (d, 0, dim));
      comp = d;
    }


    template <int dim>
    class Velocity : public MultiComponentFunction<dim>
    {
    public:
      Velocity (const double initial_time = 0.0);

      virtual double value (const Point<dim> &p,
                            const unsigned int component = 0) const;

      virtual void value_list (const std::vector<Point<dim> > &points,
                               std::vector<double>            &values,
                               const unsigned int              component = 0) const;
      virtual void set_geometry (const double height,
                                 const double width);
    private:
      double width;
      double height;
    };


    template <int dim>
    Velocity<dim>::Velocity (const double initial_time)
      :
      MultiComponentFunction<dim> (initial_time)
    {}


    template <int dim>
    void Velocity<dim>::value_list (const std::vector<Point<dim> > &points,
                                    std::vector<double> &values,
                                    const unsigned int) const
    {
      const unsigned int n_points = points.size();
      Assert (values.size() == n_points,
              ExcDimensionMismatch (values.size(), n_points));
      for (unsigned int i = 0; i < n_points; ++i)
        values[i] = Velocity<dim>::value (points[i]);
    }


    template <int dim>
    double Velocity<dim>::value (const Point<dim> &p,
                                 const unsigned int) const
    {
      if (this->comp == 0)
        {
          const double Um = 1.5;
          if (dim == 2)
            {
              return 4.0*Um*p(1)*(width - p(1))/(width*width);
            }
          else if (dim == 3)
            {
              return 4.0*Um*p(1)*(width - p(1))*p(2)*(height - p(2))
                     /(height*height);
            }
          else
            {
              Assert (false, ExcNotImplemented());
            }
        }
      else
        return 0.;
    }

    template<int dim>
    void Velocity<dim>::set_geometry (const double height,
                                      const double width)
    {
      this->height = height;
      this->width = width;
    }

    template <int dim>
    class Pressure: public Function<dim>
    {
    public:
      Pressure (const double initial_time = 0.0);

      virtual double value (const Point<dim> &p,
                            const unsigned int component = 0) const;

      virtual void value_list (const std::vector< Point<dim> > &points,
                               std::vector<double> &values,
                               const unsigned int component = 0) const;
    };

    template <int dim>
    Pressure<dim>::Pressure (const double initial_time)
      :
      Function<dim> (1, initial_time)
    {}


    template <int dim>
    double Pressure<dim>::value (const Point<dim> &p,
                                 const unsigned int) const
    {
      return 25.-p(0);
    }

    template <int dim>
    void Pressure<dim>::value_list (const std::vector<Point<dim> > &points,
                                    std::vector<double> &values,
                                    const unsigned int) const
    {
      const unsigned int n_points = points.size();
      Assert (values.size() == n_points, ExcDimensionMismatch (values.size(),
                                                               n_points));
      for (unsigned int i = 0; i < n_points; ++i)
        values[i] = Pressure<dim>::value (points[i]);
    }
  }


  template <int dim>
  class NavierStokesProjection
  {
  public:
    NavierStokesProjection (const RunTimeParameters::Data_Storage &data);

    void run (const bool         verbose    = false,
              const unsigned int n_plots = 10);
  protected:
    RunTimeParameters::MethodFormulation type;

    const unsigned int deg;
    const double       dt;
    const double       t_0, T, Re;
    std::string load_checkpoint_file_name;
    std::string save_checkpoint_file_name;

    EquationData::Velocity<dim>               vel_exact;
    std::map<types::global_dof_index, double> boundary_values;
    std::vector<types::boundary_id>           boundary_ids;

    Triangulation<dim> triangulation;

    FE_Q<dim> fe_velocity;
    FE_Q<dim> fe_pressure;

    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_pressure;

    std::unique_ptr<FiniteElement<dim>> output_fe;
    DoFHandler<dim> dof_handler_output;

    QGauss<dim> quadrature_pressure;
    QGauss<dim> quadrature_velocity;

    SparsityPattern constrained_velocity_sparsity_pattern;
    SparsityPattern unconstrained_velocity_sparsity_pattern;
    SparsityPattern projection_sparsity_pattern;
    SparsityPattern pressure_sparsity_pattern;
    SparsityPattern sparsity_pattern_pres_vel;
    SparsityPattern sparsity_pattern_vel_pres;

    ConstraintMatrix                  velocity_wall_constraints;
    std::array<ConstraintMatrix, dim> velocity_flow_constraints;
    ConstraintMatrix                  projection_constraints;

    SparseMatrix<double> vel_it_matrix[dim];
    SparseMatrix<double> vel_Mass;
    SparseMatrix<double> vel_Advection;
    SparseMatrix<double> pres_Diff[dim];

    Vector<double>      pres_n;
    Vector<double>      pres_n_minus_1;
    Vector<double>      phi_n;
    Vector<double>      phi_n_minus_1;
    BlockVector<double> u_n;
    BlockVector<double> u_n_minus_1;
    BlockVector<double> u_star;
    BlockVector<double> force;
    BlockVector<double> v_tmp;
    Vector<double>      pres_tmp;
    Vector<double>      rot_u;

    SparseILU<double>   prec_velocity[dim];
    SparseDirectUMFPACK prec_vel_mass;

    double height;
    double width;

    TimerOutput timer_output;

#ifdef STEP_35A_USE_PETSC
    PETScWrappers::MPI::SparseMatrix pres_Laplace;
    PETScWrappers::MPI::SparseMatrix pres_Mass;
    std::unique_ptr<PETScWrappers::PreconditionBoomerAMG> prec_pres;
    std::unique_ptr<PETScWrappers::PreconditionBoomerAMG> prec_pres_Mass;
#else
    SparseMatrix<double> pres_Laplace;
    SparseMatrix<double> pres_Mass;
    SparseILU<double> prec_pres;
    SparseDirectUMFPACK prec_pres_Mass;
#endif


    DeclException2 (ExcInvalidTimeStep,
                    double, double,
                    << " The time step " << arg1 << " is out of range."
                    << std::endl
                    << " The permitted range is (0," << arg2 << "]");

    void create_triangulation_and_dofs (const unsigned int n_refines);

    void initialize();

    void load_checkpoint();

    void save_checkpoint();

    void interpolate_velocity ();

    void diffusion_step (const bool reinit_prec);

    void projection_step ();

    void update_pressure ();

  private:
    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    bool vorticity_preconditioner_assembled;
    std::vector<XDMFEntry> xdmf_entries;
    bool                   write_mesh;

    void initialize_constraints();

    void initialize_velocity_matrices();

    void initialize_pressure_matrices();

    typedef std::tuple< typename DoFHandler<dim>::active_cell_iterator,
            typename DoFHandler<dim>::active_cell_iterator
            > IteratorTuple;

    typedef SynchronousIterators<IteratorTuple> IteratorPair;

    void initialize_gradient_operator();

    struct InitGradPerTaskData
    {
      unsigned int              d;
      unsigned int              vel_dpc;
      unsigned int              pres_dpc;
      FullMatrix<double>        local_grad;
      std::vector<types::global_dof_index> vel_local_dof_indices;
      std::vector<types::global_dof_index> pres_local_dof_indices;

      InitGradPerTaskData (const unsigned int dd,
                           const unsigned int vdpc,
                           const unsigned int pdpc)
        :
        d(dd),
        vel_dpc (vdpc),
        pres_dpc (pdpc),
        local_grad (vdpc, pdpc),
        vel_local_dof_indices (vdpc),
        pres_local_dof_indices (pdpc)
      {}
    };

    struct InitGradScratchData
    {
      unsigned int  nqp;
      FEValues<dim> fe_val_vel;
      FEValues<dim> fe_val_pres;
      InitGradScratchData (const FE_Q<dim> &fe_v,
                           const FE_Q<dim> &fe_p,
                           const QGauss<dim> &quad,
                           const UpdateFlags flags_v,
                           const UpdateFlags flags_p)
        :
        nqp (quad.size()),
        fe_val_vel (fe_v, quad, flags_v),
        fe_val_pres (fe_p, quad, flags_p)
      {}
      InitGradScratchData (const InitGradScratchData &data)
        :
        nqp (data.nqp),
        fe_val_vel (data.fe_val_vel.get_fe(),
                   data.fe_val_vel.get_quadrature(),
                   data.fe_val_vel.get_update_flags()),
        fe_val_pres (data.fe_val_pres.get_fe(),
                    data.fe_val_pres.get_quadrature(),
                    data.fe_val_pres.get_update_flags())
      {}
    };

    void assemble_one_cell_of_gradient (const IteratorPair  &SI,
                                        InitGradScratchData &scratch,
                                        InitGradPerTaskData &data);

    void copy_gradient_local_to_global (const InitGradPerTaskData &data);

    void assemble_advection_term();

    struct AdvectionPerTaskData
    {
      FullMatrix<double>        local_advection;
      Vector<double>            dummy_cell_vector;
      std::vector<types::global_dof_index> local_dof_indices;
      AdvectionPerTaskData (const unsigned int dpc)
        :
        local_advection (dpc, dpc),
        dummy_cell_vector(dpc),
        local_dof_indices (dpc)
      {}
    };

    struct AdvectionScratchData
    {
      unsigned int                 nqp;
      unsigned int                 dpc;
      std::vector< Tensor<1,dim> > u_star_local;
      std::vector< Tensor<1,dim> > grad_u_star;
      std::vector<double>          u_star_tmp;
      FEValues<dim>                fe_val;
      AdvectionScratchData (const FE_Q<dim> &fe,
                            const QGauss<dim> &quad,
                            const UpdateFlags flags)
        :
        nqp (quad.size()),
        dpc (fe.dofs_per_cell),
        u_star_local (nqp),
        grad_u_star (nqp),
        u_star_tmp (nqp),
        fe_val (fe, quad, flags)
      {}

      AdvectionScratchData (const AdvectionScratchData &data)
        :
        nqp (data.nqp),
        dpc (data.dpc),
        u_star_local (nqp),
        grad_u_star (nqp),
        u_star_tmp (nqp),
        fe_val (data.fe_val.get_fe(),
               data.fe_val.get_quadrature(),
               data.fe_val.get_update_flags())
      {}
    };

    void assemble_one_cell_of_advection
    (const typename DoFHandler<dim>::active_cell_iterator &cell,
     AdvectionScratchData &scratch,
     AdvectionPerTaskData &data);

    void copy_advection_local_to_global (const AdvectionPerTaskData &data);

    void diffusion_component_solve (const unsigned int d);

    void output_results (const unsigned int step, const double time);

    void assemble_vorticity ();
  };


  template <int dim>
  NavierStokesProjection<dim>::NavierStokesProjection
  (const RunTimeParameters::Data_Storage &data)
    :
    type (data.form),
    deg (data.pressure_degree),
    dt (data.dt),
    t_0 (data.initial_time),
    T (data.final_time),
    Re (data.Reynolds),
    load_checkpoint_file_name (data.load_checkpoint_file_name),
    save_checkpoint_file_name (data.save_checkpoint_file_name),
    vel_exact (data.initial_time),
    fe_velocity (deg+1),
    fe_pressure (deg),
    dof_handler_velocity (triangulation),
    dof_handler_pressure (triangulation),
    dof_handler_output (triangulation),
    quadrature_pressure (deg+1),
    quadrature_velocity (deg+2),
    u_n(dim),
    u_n_minus_1(dim),
    u_star(dim),
    force(dim),
    v_tmp(dim),
    timer_output (std::cout, TimerOutput::OutputFrequency::summary, TimerOutput::cpu_and_wall_times),
    vel_max_its (data.vel_max_iterations),
    vel_Krylov_size (data.vel_Krylov_size),
    vel_off_diagonals (data.vel_off_diagonals),
    vel_update_prec (data.vel_update_prec),
    vel_eps (data.vel_eps),
    vel_diag_strength (data.vel_diag_strength),
    vorticity_preconditioner_assembled(false),
    xdmf_entries(),
    write_mesh(true)
  {
    if (deg < 1)
      std::cout << " WARNING: The chosen pair of finite element spaces is not stable."
                << std::endl
                << " The obtained results will be nonsense"
                << std::endl;

    AssertThrow (!  ( (dt <= 0.) || (dt > .5*T)), ExcInvalidTimeStep (dt, .5*T));

    create_triangulation_and_dofs (data.n_global_refines);
    initialize();
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::
  create_triangulation_and_dofs (const unsigned int n_refines)
  {
    TimerOutput::Scope timer_scope(timer_output, "create_triangulation_and_dofs");
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (triangulation);
    {
      // std::string filename = "nsbench2.inp";
      std::string filename = "cylinderextruded.msh";
      std::ifstream file (filename.c_str());
      Assert (file, ExcFileNotOpen (filename.c_str()));
      grid_in.read_msh (file);
      // grid_in.read_ucd (file);
    }
    std::cout << "Number of refines = " << n_refines
              << '\n';
    triangulation.refine_global (n_refines);
    std::cout << "minimum cell diameter: "
              << GridTools::minimal_cell_diameter (triangulation)
              << '\n';
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;
    boundary_ids = triangulation.get_boundary_ids();

    dof_handler_velocity.distribute_dofs (fe_velocity);
    DoFRenumbering::boost::Cuthill_McKee (dof_handler_velocity);
    dof_handler_pressure.distribute_dofs (fe_pressure);
    DoFRenumbering::boost::Cuthill_McKee (dof_handler_pressure);
    // Unfortunately, the FESystem constructor depends on the dimensionality, so
    // (without a reinit function) we must use dynamic allocation.
    if (dim == 2)
      {
        output_fe = std::unique_ptr<FiniteElement<dim>>
          (new FESystem<dim> (fe_velocity, dim, fe_pressure, 1, fe_velocity, 1));
      }
    else
      {
        output_fe = std::unique_ptr<FiniteElement<dim>>
          (new FESystem<dim> (fe_velocity, dim, fe_pressure, 1));
      }
    dof_handler_output.distribute_dofs (*output_fe);
#ifdef DEBUG
    const int add_vorticity = (dim == 2) ? 1 : 0;
    Assert (dof_handler_output.n_dofs() ==
            ((dim + add_vorticity)*dof_handler_velocity.n_dofs() +
             dof_handler_pressure.n_dofs()),
            ExcInternalError());
#endif

    // determine the maximum z coordinate.
    for (const auto ti : triangulation.active_cell_iterators())
      {
        for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
          {
            if (dim == 3)
              height = std::max(ti->vertex(i)[dim - 1], height);
            width = std::max(ti->vertex(i)[1], width);
          }
      }
    vel_exact.set_geometry (height, width);

    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;

    initialize_constraints();
    initialize_velocity_matrices();
    initialize_pressure_matrices();
    initialize_gradient_operator();

    pres_n.reinit (dof_handler_pressure.n_dofs());
    pres_n_minus_1.reinit (dof_handler_pressure.n_dofs());
    phi_n.reinit (dof_handler_pressure.n_dofs());
    phi_n_minus_1.reinit (dof_handler_pressure.n_dofs());
    pres_tmp.reinit (dof_handler_pressure.n_dofs());
    for (unsigned int d = 0; d < dim; ++d)
      {
        u_n.block(d).reinit (dof_handler_velocity.n_dofs());
        u_n_minus_1.block(d).reinit (dof_handler_velocity.n_dofs());
        u_star.block(d).reinit (dof_handler_velocity.n_dofs());
        force.block(d).reinit (dof_handler_velocity.n_dofs());
        v_tmp.block(d).reinit (dof_handler_velocity.n_dofs());
      }
    rot_u.reinit (dof_handler_velocity.n_dofs());

    std::cout << "dim (X_h) = " << (dof_handler_velocity.n_dofs()*dim)
              << std::endl
              << "dim (M_h) = " << dof_handler_pressure.n_dofs()
              << std::endl
              << "Re        = " << Re
              << std::endl
              << std::endl;
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::initialize()
  {
    EquationData::Pressure<dim> pres (t_0);
    VectorTools::interpolate (dof_handler_pressure, pres, pres_n_minus_1);
    pres.advance_time (dt);
    VectorTools::interpolate (dof_handler_pressure, pres, pres_n);
    phi_n = 0.;
    phi_n_minus_1 = 0.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        vel_exact.set_time (t_0);
        vel_exact.set_component(d);
        VectorTools::interpolate
        (dof_handler_velocity, ZeroFunction<dim>(), u_n_minus_1.block(d));
        vel_exact.advance_time (dt);
        VectorTools::interpolate
        (dof_handler_velocity, ZeroFunction<dim>(), u_n.block(d));
      }
  }

  template<int dim>
  void
  NavierStokesProjection<dim>::load_checkpoint()
  {
    TimerOutput::Scope timer_scope(timer_output, "load_checkpoint");
    if (load_checkpoint_file_name.size() == 0)
      {
        return;
      }
    for (unsigned int block_n = 0; block_n < dim; ++block_n)
      {
        H5::load(load_checkpoint_file_name, "u_n" + Utilities::int_to_string(block_n),
                 u_n.block(block_n));
        H5::load(load_checkpoint_file_name, "u_n_minus_1" + Utilities::int_to_string(block_n),
                 u_n_minus_1.block(block_n));
      }
    H5::load(load_checkpoint_file_name, std::string("phi_n"), phi_n);
    H5::load(load_checkpoint_file_name, std::string("phi_n_minus_1"), phi_n_minus_1);
    H5::load(load_checkpoint_file_name, std::string("pres_n"), pres_n);
    H5::load(load_checkpoint_file_name, std::string("pres_n_minus_1"), pres_n_minus_1);
  }


  template<int dim>
  void
  NavierStokesProjection<dim>::save_checkpoint()
  {
    TimerOutput::Scope timer_scope(timer_output, "save_checkpoint");
    if (save_checkpoint_file_name.size() == 0)
      {
        return;
      }
    std::cout << "outfile name is '" << save_checkpoint_file_name << "'" << std::endl;
    for (unsigned int block_n = 0; block_n < dim; ++block_n)
      {
        H5::save(save_checkpoint_file_name, "u_n" + Utilities::int_to_string(block_n),
                 u_n.block(block_n));
        H5::save(save_checkpoint_file_name, "u_n_minus_1" + Utilities::int_to_string(block_n),
                 u_n_minus_1.block(block_n));
      }
    H5::save(save_checkpoint_file_name, std::string("phi_n"), phi_n);
    H5::save(save_checkpoint_file_name, std::string("phi_n_minus_1"), phi_n_minus_1);
    H5::save(save_checkpoint_file_name, std::string("pres_n"), pres_n);
    H5::save(save_checkpoint_file_name, std::string("pres_n_minus_1"), pres_n_minus_1);
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::initialize_constraints()
  {
    // check the boundary ids
#ifdef DEBUG
    for (const types::boundary_id boundary_id : boundary_ids)
      {
        Assert(boundary_id == 1 || boundary_id == 2 ||
               boundary_id == 3 || boundary_id == 4,
               ExcMessage("unknown boundary id"));
      }
#endif

    for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
    {
      vel_exact.set_component(dim_n);
      // different components have different inflow conditions
      VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                               2,
                                               vel_exact,
                                               velocity_flow_constraints[dim_n]);

      // we do not enforce zero outflow on the first velocity component
      if (dim_n != 0)
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 3,
                                                 ZeroFunction<dim>(),
                                                 velocity_flow_constraints[dim_n]);
      velocity_flow_constraints[dim_n].close();
    }

    {
      VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                               1,
                                               ZeroFunction<dim>(),
                                               velocity_wall_constraints);

      VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                               4,
                                               ZeroFunction<dim>(),
                                               velocity_wall_constraints);
      velocity_wall_constraints.close();
    }

    // constraints for the projection term
    {
      // TODO is this really the only projection boundary value (zero
      // at outflow)?
      VectorTools::interpolate_boundary_values(dof_handler_pressure,
                                               3,
                                               ZeroFunction<dim>(),
                                               projection_constraints);
      projection_constraints.close();
    }
  }

  template <int dim>
  void
  NavierStokesProjection<dim>::initialize_velocity_matrices()
  {
    {
      DynamicSparsityPattern dynamic_sparsity_pattern
        (dof_handler_velocity.n_dofs(), dof_handler_velocity.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_velocity,
                                      dynamic_sparsity_pattern);
      unconstrained_velocity_sparsity_pattern.copy_from (dynamic_sparsity_pattern);
    }

    {
      DynamicSparsityPattern dynamic_sparsity_pattern
        (dof_handler_velocity.n_dofs(), dof_handler_velocity.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_velocity,
                                      dynamic_sparsity_pattern,
                                      velocity_wall_constraints,
                                      /*keep_constrained_dofs=*/false);
      constrained_velocity_sparsity_pattern.copy_from (dynamic_sparsity_pattern);
    }

    for (unsigned int d = 0; d < dim; ++d)
      vel_it_matrix[d].reinit (constrained_velocity_sparsity_pattern);
    vel_Advection.reinit (constrained_velocity_sparsity_pattern);

    vel_Mass.reinit (unconstrained_velocity_sparsity_pattern);
    MatrixCreator::create_mass_matrix (dof_handler_velocity,
                                       quadrature_velocity,
                                       vel_Mass);
  }

  template <int dim>
  void
  NavierStokesProjection<dim>::initialize_pressure_matrices()
  {
    {
      DynamicSparsityPattern dynamic_sparsity_pattern
        (dof_handler_pressure.n_dofs(), dof_handler_pressure.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_pressure,
                                      dynamic_sparsity_pattern,
                                      projection_constraints,
                                      /*keep_constrained_dofs=*/false);
      projection_sparsity_pattern.copy_from (dynamic_sparsity_pattern);
    }

    {
      DynamicSparsityPattern dynamic_sparsity_pattern
        (dof_handler_pressure.n_dofs(), dof_handler_pressure.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_pressure,
                                      dynamic_sparsity_pattern);
      pressure_sparsity_pattern.copy_from (dynamic_sparsity_pattern);
    }

    // the pressure/projection matrices never change, so set up
    // preconditioners here too
    const IndexSet pressure_dofs = complete_index_set(dof_handler_pressure.n_dofs());
#ifdef STEP_35A_USE_PETSC
    pres_Laplace.reinit(pressure_dofs, pressure_dofs, projection_sparsity_pattern,
                        MPI_COMM_WORLD);
    pres_Mass.reinit (pressure_dofs, pressure_dofs, pressure_sparsity_pattern,
                      MPI_COMM_WORLD);
#else
    pres_Laplace.reinit (projection_sparsity_pattern);
    pres_Mass.reinit (pressure_sparsity_pattern);
#endif
    {
      FEValues<dim> fe_values(fe_pressure,
                              quadrature_pressure,
                              update_values | update_gradients | update_JxW_values);
      FullMatrix<double> cell_laplace_matrix(fe_pressure.dofs_per_cell,
                                             fe_pressure.dofs_per_cell);
      FullMatrix<double> cell_mass_matrix(fe_pressure.dofs_per_cell,
                                          fe_pressure.dofs_per_cell);
      std::vector<types::global_dof_index> cell_dofs(fe_pressure.dofs_per_cell);
      for (const auto cell : dof_handler_pressure.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(cell_dofs);
          cell_mass_matrix = 0.0;
          cell_laplace_matrix = 0.0;
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              for (unsigned int i = 0; i < fe_pressure.dofs_per_cell; ++i)
                for (unsigned int j = 0; j < fe_pressure.dofs_per_cell; ++j)
                  {
                    cell_laplace_matrix(i, j) += (fe_values.shape_grad(i, q)
                                                  * fe_values.shape_grad(j, q))
                      * fe_values.JxW(q);
                    cell_mass_matrix(i, j) += (fe_values.shape_value(i, q)
                                               * fe_values.shape_value(j, q))
                      * fe_values.JxW(q);
                  }
            }
          projection_constraints.distribute_local_to_global(cell_laplace_matrix,
                                                            cell_dofs,
                                                            pres_Laplace);
          pres_Mass.add(cell_dofs, cell_mass_matrix);
        }
#ifdef STEP_35A_USE_PETSC
      pres_Laplace.compress(VectorOperation::add);
      pres_Mass.compress(VectorOperation::add);
      PETScWrappers::PreconditionBoomerAMG::AdditionalData options;
      options.symmetric_operator = true;

      prec_pres.reset(new PETScWrappers::PreconditionBoomerAMG(pres_Laplace,
                                                               options));
      prec_pres_Mass.reset(new PETScWrappers::PreconditionBoomerAMG(pres_Mass,
                                                                    options));
#else
      prec_pres.initialize(pres_Laplace,
                           SparseILU<double>::
                           AdditionalData(vel_diag_strength,
                                          vel_off_diagonals,
                                          /*use_previous_sparsity=*/true));
      prec_pres_Mass.initialize(pres_Mass);
#endif
    }
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::initialize_gradient_operator()
  {
    {
      DynamicSparsityPattern dynamic_sparsity_pattern
        (dof_handler_velocity.n_dofs(), dof_handler_pressure.n_dofs());
      DoFTools::make_sparsity_pattern (dof_handler_velocity,
                                       dof_handler_pressure,
                                       dynamic_sparsity_pattern);
      sparsity_pattern_pres_vel.copy_from (dynamic_sparsity_pattern);
    }

    {
      DynamicSparsityPattern dynamic_sparsity_pattern
        (dof_handler_pressure.n_dofs(), dof_handler_velocity.n_dofs());
      DoFTools::make_sparsity_pattern (dof_handler_pressure,
                                       dof_handler_velocity,
                                       dynamic_sparsity_pattern);
      sparsity_pattern_vel_pres.copy_from (dynamic_sparsity_pattern);
    }

    InitGradPerTaskData per_task_data (0, fe_velocity.dofs_per_cell,
                                       fe_pressure.dofs_per_cell);
    InitGradScratchData scratch_data (fe_velocity,
                                      fe_pressure,
                                      quadrature_velocity,
                                      update_gradients | update_JxW_values,
                                      update_values);

    for (unsigned int d = 0; d < dim; ++d)
      {
        pres_Diff[d].reinit (sparsity_pattern_pres_vel);
        per_task_data.d = d;
        WorkStream::run (IteratorPair
                         (IteratorTuple (dof_handler_velocity.begin_active(),
                                         dof_handler_pressure.begin_active())),
                         IteratorPair (IteratorTuple (dof_handler_velocity.end(),
                                                      dof_handler_pressure.end())),
                         *this,
                         &NavierStokesProjection<dim>::assemble_one_cell_of_gradient,
                         &NavierStokesProjection<dim>::copy_gradient_local_to_global,
                         scratch_data,
                         per_task_data
                        );
      }
  }

  template <int dim>
  void
  NavierStokesProjection<dim>::
  assemble_one_cell_of_gradient (const IteratorPair  &SI,
                                 InitGradScratchData &scratch,
                                 InitGradPerTaskData &data)
  {
    scratch.fe_val_vel.reinit (std::get<0> (*SI));
    scratch.fe_val_pres.reinit (std::get<1> (*SI));

    std::get<0> (*SI)->get_dof_indices (data.vel_local_dof_indices);
    std::get<1> (*SI)->get_dof_indices (data.pres_local_dof_indices);

    data.local_grad = 0.;
    for (unsigned int q = 0; q < scratch.nqp; ++q)
      {
        for (unsigned int i = 0; i < data.vel_dpc; ++i)
          for (unsigned int j = 0; j < data.pres_dpc; ++j)
            data.local_grad (i, j) += -scratch.fe_val_vel.JxW(q) *
                                      scratch.fe_val_vel.shape_grad (i, q)[data.d] *
                                      scratch.fe_val_pres.shape_value (j, q);
      }
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::
  copy_gradient_local_to_global(const InitGradPerTaskData &data)
  {
    for (unsigned int i = 0; i < data.vel_dpc; ++i)
      for (unsigned int j = 0; j < data.pres_dpc; ++j)
        pres_Diff[data.d].add
        (data.vel_local_dof_indices[i], data.pres_local_dof_indices[j],
         data.local_grad (i, j));
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::run (const bool verbose,
                                    const unsigned int output_interval)
  {
    ConditionalOStream verbose_cout (std::cout, verbose);

    const unsigned int n_steps = static_cast<unsigned int>((T - t_0)/dt);
    vel_exact.set_time (t_0 + 2.*dt);
    vel_exact.set_geometry (height, width);
    unsigned int step_start_n = 2;
    unsigned int step_start_offset_n = 2;
    if (load_checkpoint_file_name.size() == 0)
      {
        output_results(1, t_0);
      }
    else
      {
        load_checkpoint();
        step_start_n = static_cast<unsigned int>(t_0/dt);
        step_start_offset_n = 0;
      }

    for (unsigned int n = step_start_n;
         n <= step_start_n + n_steps - step_start_offset_n; ++n)
      {
        double current_time = (t_0 + (n - step_start_n + step_start_offset_n)*dt);
        if (n % output_interval == 0)
          {
            verbose_cout << "Plotting Solution" << std::endl;
            output_results(n, current_time);
          }
        std::cout << "Step = " << n << " Time = "
                  << current_time << std::endl;
        verbose_cout << "  Interpolating the velocity " << std::endl;

        // both of these are needed to do the diffusion step
        if (n == step_start_offset_n)
          {
            interpolate_velocity();
            assemble_advection_term();
          }
        verbose_cout << "  Diffusion Step" << std::endl;
        if (n % vel_update_prec == 0)
          verbose_cout << "    With reinitialization of the preconditioner"
                       << std::endl;
        diffusion_step ((n % vel_update_prec == 0) || (n == step_start_n));

        interpolate_velocity();
        // we may update the pressure while we assemble the advection matrix
        // (both are expensive)
        // assemble_advection_term();
        {
          Threads::TaskGroup<void> tasks;
          tasks += Threads::new_task
            ([this]()
             {
               assemble_advection_term();
             });

          tasks += Threads::new_task
            ([this, n, step_start_n, &verbose_cout]()
             {
               verbose_cout << "  Projection Step" << std::endl;
               projection_step();
               verbose_cout << "  Updating the Pressure" << std::endl;
               update_pressure ();
             });
        }

        vel_exact.advance_time(dt);

      }
    save_checkpoint();
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::interpolate_velocity()
  {
    TimerOutput::Scope timer_scope(timer_output, "interpolate_velocity");
    for (unsigned int d = 0; d < dim; ++d)
      {
        u_star.block(d).equ(2.0, u_n.block(d));
        u_star.block(d).add(-1.0, u_n_minus_1.block(d));
      }
  }


  // The implementation of a diffusion step. Note that the expensive operation
  // is the diffusion solve at the end of the function, which we have to do
  // once for each velocity component. To accelerate things a bit, we allow
  // to do this in %parallel, using the Threads::new_task function which makes
  // sure that the <code>dim</code> solves are all taken care of and are
  // scheduled to available processors: if your machine has more than one
  // processor core and no other parts of this program are using resources
  // currently, then the diffusion solves will run in %parallel. On the other
  // hand, if your system has only one processor core then running things in
  // %parallel would be inefficient (since it leads, for example, to cache
  // congestion) and things will be executed sequentially.
  template <int dim>
  void
  NavierStokesProjection<dim>::diffusion_step (const bool reinit_prec)
  {
    pres_tmp.equ(-1.0, pres_n);
    pres_tmp.add(-4.0/3.0, phi_n, 1.0/3.0, phi_n_minus_1);

    {
      TimerOutput::Scope timer_scope(timer_output, "setup_diffusion_rhs");
      Threads::TaskGroup<void> tasks;
      for (unsigned int d = 0; d < dim; ++d)
        {
          // copy the vectors...
          tasks += Threads::new_task
            ([this, d]()
             {
               force.block(d) = 0.;
               v_tmp.block(d).equ(2.0/dt, u_n.block(d));
               v_tmp.block(d).add(-0.5/dt, u_n_minus_1.block(d));
               vel_Mass.vmult_add (force.block(d), v_tmp.block(d));

               pres_Diff[d].vmult_add (force.block(d), pres_tmp);
               u_n_minus_1.block(d) = u_n.block(d);
             }
             );
          // and the matrices.
          tasks += Threads::new_task
            ([this, d]()
             {
               vel_it_matrix[d].copy_from (vel_Advection);
             }
             );
        }
    }

    {
      TimerOutput::Scope timer_scope(timer_output, "solve_diffusion_component");
      Threads::TaskGroup<void> tasks;
      for (unsigned int d = 0; d < dim; ++d)
        {
          tasks += Threads::new_task
            ([this, d, reinit_prec]()
             {
               velocity_flow_constraints[d].condense(vel_it_matrix[d], force.block(d));
               if (reinit_prec)
                 {
                   this->prec_velocity[d].initialize (this->vel_it_matrix[d],
                                                      SparseILU<double>::
                                                      AdditionalData (this->vel_diag_strength,
                                                                      this->vel_off_diagonals,
                                                                      /*use_previous_sparsity=*/true));
                 }
               this->diffusion_component_solve(d);

               velocity_flow_constraints[d].distribute(u_n.block(d));
               velocity_wall_constraints.distribute(u_n.block(d));
             });
        }
    }
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::diffusion_component_solve (const unsigned int d)
  {
    SolverControl solver_control (vel_max_its, vel_eps*force.block(d).l2_norm());
    SolverGMRES<> gmres (solver_control,
                         SolverGMRES<>::AdditionalData (vel_Krylov_size));
    gmres.solve (vel_it_matrix[d], u_n.block(d), force.block(d), prec_velocity[d]);
  }


  // The following few functions deal with assembling the advection terms,
  // which is the part of the system matrix for the diffusion step that
  // changes at every time step. As mentioned above, we will run the assembly
  // loop over all cells in %parallel, using the WorkStream class and other
  // facilities as described in the documentation module on @ref threads.
  template <int dim>
  void
  NavierStokesProjection<dim>::assemble_advection_term()
  {
    TimerOutput::Scope timer_scope(timer_output, "assemble_advection_term");
    vel_Advection = 0.;
    AdvectionPerTaskData data (fe_velocity.dofs_per_cell);
    AdvectionScratchData scratch (fe_velocity, quadrature_velocity,
                                  update_values |
                                  update_JxW_values |
                                  update_gradients);
    WorkStream::run (dof_handler_velocity.begin_active(),
                     dof_handler_velocity.end(), *this,
                     &NavierStokesProjection<dim>::assemble_one_cell_of_advection,
                     &NavierStokesProjection<dim>::copy_advection_local_to_global,
                     scratch,
                     data);
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::
  assemble_one_cell_of_advection
  (const typename DoFHandler<dim>::active_cell_iterator &cell,
   AdvectionScratchData &scratch,
   AdvectionPerTaskData &data)
  {
    scratch.fe_val.reinit(cell);
    cell->get_dof_indices (data.local_dof_indices);
    for (unsigned int d = 0; d < dim; ++d)
      {
        scratch.fe_val.get_function_values (u_star.block(d), scratch.u_star_tmp);
        for (unsigned int q = 0; q < scratch.nqp; ++q)
          scratch.u_star_local[q][d] = scratch.u_star_tmp[q];
      }

    for (unsigned int d = 0; d < dim; ++d)
      {
        scratch.fe_val.get_function_gradients (u_star.block(d), scratch.grad_u_star);
        for (unsigned int q = 0; q < scratch.nqp; ++q)
          {
            if (d==0)
              scratch.u_star_tmp[q] = 0.;
            scratch.u_star_tmp[q] += scratch.grad_u_star[q][d];
          }
      }

    const double time_step_term = 1.5/dt;
    const double inverse_re = 1.0/Re;

    for (unsigned int q = 0; q < scratch.nqp; ++q)
      {
        scratch.u_star_tmp[q] *= 0.5;
        // mass contribution
        scratch.u_star_tmp[q] += time_step_term;
      }

    data.local_advection = 0.;
    for (unsigned int q = 0; q < scratch.nqp; ++q)
      for (unsigned int i = 0; i < scratch.dpc; ++i)
        for (unsigned int j = 0; j < scratch.dpc; ++j)
          data.local_advection(i, j) +=
            (
             // diffusion
             inverse_re *
             (scratch.fe_val.shape_grad(i, q) * scratch.fe_val.shape_grad(j, q))
             +
             scratch.fe_val.shape_value (i, q)
             * (// advection
                scratch.u_star_local[q] *
                scratch.fe_val.shape_grad (j, q)
                +
                // note that the mass contribution is here
                scratch.u_star_tmp[q] *
                scratch.fe_val.shape_value (j, q))
             )
            * scratch.fe_val.JxW(q);
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::
  copy_advection_local_to_global(const AdvectionPerTaskData &data)
  {
    Assert(data.local_dof_indices.size() == fe_velocity.dofs_per_cell,
           ExcMessage("dimension mismatch"));
    for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
      {
        if (dim_n == 0)
          {
            velocity_wall_constraints.distribute_local_to_global
              (data.local_advection,
               data.dummy_cell_vector,
               data.local_dof_indices,
               vel_Advection,
               force.block(0));
          }
        else
          {
            velocity_wall_constraints.distribute_local_to_global(data.dummy_cell_vector,
                                                                 data.local_dof_indices,
                                                                 force.block(dim_n),
                                                                 data.local_advection);
          }
      }
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::projection_step ()
  {
    {
      TimerOutput::Scope timer_scope(timer_output, "projection_step_setup");
      pres_Diff[0].Tvmult (pres_tmp, u_n.block(0));
      for (unsigned d = 1; d < dim; ++d)
        pres_Diff[d].Tvmult_add (pres_tmp, u_n.block(d));

      phi_n_minus_1 = phi_n;
    }

#ifdef STEP_35A_USE_PETSC
    {
      const IndexSet pressure_dofs = complete_index_set(dof_handler_pressure.n_dofs());
      PETScWrappers::MPI::Vector petsc_solution(pressure_dofs, MPI_COMM_WORLD);
      PETScWrappers::MPI::Vector petsc_rhs(pressure_dofs, MPI_COMM_WORLD);

      {
        TimerOutput::Scope timer_scope(timer_output, "projection_step_setup");
        copy_to_petsc(pres_tmp, petsc_rhs);
      }

      {
        TimerOutput::Scope timer_scope(timer_output, "projection_step_solve");
        SolverControl solver_control (vel_max_its, vel_eps*pres_tmp.l2_norm());
        PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);
        solver.solve(pres_Laplace, petsc_solution, petsc_rhs, *prec_pres);
      }

      {
        TimerOutput::Scope timer_scope(timer_output, "projection_step_setup");
        copy_from_petsc(phi_n, petsc_solution);
        projection_constraints.distribute(phi_n);
      }
    }
#else
    {
      TimerOutput::Scope timer_scope(timer_output, "projection_step_solve");
      SolverControl solver_control (vel_max_its, vel_eps*pres_tmp.l2_norm());
      SolverCG<> cg (solver_control);
      projection_constraints.condense(pres_tmp);
      cg.solve (pres_Laplace, phi_n, pres_tmp, prec_pres);
      projection_constraints.distribute(phi_n);
    }
#endif

    {
      TimerOutput::Scope timer_scope(timer_output, "projection_step_setup");
      phi_n *= 1.5/dt;
    }
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::update_pressure ()
  {
    TimerOutput::Scope timer_scope(timer_output, "update_pressure");
    pres_n_minus_1 = pres_n;
    switch (type)
      {
      case RunTimeParameters::METHOD_STANDARD:
        pres_n += phi_n;
        break;
      case RunTimeParameters::METHOD_ROTATIONAL:
        pres_n = pres_tmp;
#ifdef STEP_35A_USE_PETSC
        {
          const IndexSet pressure_dofs = complete_index_set(dof_handler_pressure.n_dofs());
          PETScWrappers::MPI::Vector petsc_solution(pressure_dofs, MPI_COMM_WORLD);
          PETScWrappers::MPI::Vector petsc_rhs(pressure_dofs, MPI_COMM_WORLD);
          copy_to_petsc(pres_n, petsc_rhs);

          SolverControl solver_control (vel_max_its, vel_eps*pres_tmp.l2_norm());
          PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);
          solver.solve(pres_Mass, petsc_solution, petsc_rhs, *prec_pres_Mass);
          copy_from_petsc(pres_n, petsc_solution);
        }
#else
        prec_pres_Mass.solve (pres_n);
#endif
        pres_n.sadd(1.0/Re, 1.0, pres_n_minus_1);
        pres_n.add(1.0, phi_n);
        break;
      default:
        Assert (false, ExcNotImplemented());
      };
  }


  template <int dim>
  void NavierStokesProjection<dim>::output_results (const unsigned int step, const double time)
  {
    TimerOutput::Scope timer_scope(timer_output, "output_results");
    int add_vorticity = (dim == 2) ? 1 : 0;
    std::vector<std::string> joint_solution_names (dim, "v");
    joint_solution_names.push_back ("p");
    if (add_vorticity)
      {
        assemble_vorticity ();
        joint_solution_names.push_back ("rot_u");
      }
    const auto &joint_fe = *output_fe;

    static Vector<double> joint_solution (dof_handler_output.n_dofs());
    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
        loc_vel_dof_indices (fe_velocity.dofs_per_cell),
        loc_pres_dof_indices (fe_pressure.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
    joint_cell = dof_handler_output.begin_active(),
    joint_endc = dof_handler_output.end(),
    vel_cell   = dof_handler_velocity.begin_active(),
    pres_cell  = dof_handler_pressure.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell)
      {
        joint_cell->get_dof_indices (loc_joint_dof_indices);
        vel_cell->get_dof_indices (loc_vel_dof_indices),
                 pres_cell->get_dof_indices (loc_pres_dof_indices);
        for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
          switch (joint_fe.system_to_base_index(i).first.first)
            {
            case 0:
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                u_n.block(joint_fe.system_to_base_index(i).first.second)
                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 1:
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                pres_n (loc_pres_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 2:
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                rot_u (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            default:
              Assert (false, ExcInternalError());
            }
      }
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler_output);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation (dim + 1 + add_vorticity,
                              DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim]
      = DataComponentInterpretation::component_is_scalar;
    if (add_vorticity)
      {
        component_interpretation[dim + 1]
          = DataComponentInterpretation::component_is_scalar;
      }
    data_out.add_data_vector (joint_solution,
                              joint_solution_names,
                              DataOut<dim>::type_dof_data,
                              component_interpretation);
    if (dim == 2)
      data_out.build_patches (deg + 2);
    else
      data_out.build_patches ();

    std::string h5_solution_file_name = "solution-"
                                        + Utilities::int_to_string(step, 9) + ".h5";
    std::string mesh_file_name = "mesh.h5";
    std::string xdmf_filename = "solution.xdmf";

    DataOutBase::DataOutFilter data_filter
    (DataOutBase::DataOutFilterFlags(true, true));
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, write_mesh, mesh_file_name,
                                 h5_solution_file_name, MPI_COMM_WORLD);
    // only save the triangulation, FE, and DoFHandler once
    if (write_mesh)
      {
        write_mesh = false;
        std::ofstream fe_file_stream("finite_element.txt");
        fe_file_stream << fe_velocity.get_name () << std::endl;

        for (unsigned int i = 0; i < 2; ++i)
          {
            std::string file_name;
            file_name = ((i == 0) ? "dof_handler.txt" : "triangulation.txt");
            std::filebuf file_buffer;
            file_buffer.open(file_name, std::ios::out);

            std::ostream out_stream (&file_buffer);
            boost::archive::text_oarchive archive (out_stream);
            (i == 0) ? archive << dof_handler_velocity : archive << triangulation;
          }
      }
    auto new_xdmf_entry = data_out.create_xdmf_entry
                          (data_filter, mesh_file_name, h5_solution_file_name,
                           time, MPI_COMM_WORLD);
    xdmf_entries.push_back(std::move(new_xdmf_entry));
    data_out.write_xdmf_file(xdmf_entries, xdmf_filename, MPI_COMM_WORLD);

    std::string snapshot_name = "snapshot-" + Utilities::int_to_string(step, 9)
                                + ".h5";
    H5::save_block_vector(snapshot_name, u_n);
  }


  template <int dim>
  void NavierStokesProjection<dim>::assemble_vorticity ()
  {
    TimerOutput::Scope timer_scope(timer_output, "assemble_vorticity");
    Assert (dim == 2, ExcNotImplemented());
    if (!vorticity_preconditioner_assembled)
      {
        prec_vel_mass.initialize (vel_Mass);
        vorticity_preconditioner_assembled = true;
      }

    FEValues<dim> fe_val_vel (fe_velocity, quadrature_velocity,
                              update_gradients |
                              update_JxW_values |
                              update_values);
    const unsigned int dpc = fe_velocity.dofs_per_cell,
                       nqp = quadrature_velocity.size();
    std::vector<types::global_dof_index> ldi (dpc);
    Vector<double> loc_rot (dpc);

    std::vector<Tensor<1,dim> > grad_u1 (nqp), grad_u2 (nqp);
    rot_u = 0.;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_velocity.begin_active(),
    end  = dof_handler_velocity.end();
    for (; cell != end; ++cell)
      {
        fe_val_vel.reinit (cell);
        cell->get_dof_indices (ldi);
        fe_val_vel.get_function_gradients (u_n.block(0), grad_u1);
        fe_val_vel.get_function_gradients (u_n.block(1), grad_u2);
        loc_rot = 0.;
        for (unsigned int q = 0; q < nqp; ++q)
          for (unsigned int i = 0; i < dpc; ++i)
            loc_rot(i) += (grad_u2[q][0] - grad_u1[q][1]) *
                          fe_val_vel.shape_value (i, q) *
                          fe_val_vel.JxW(q);

        for (unsigned int i = 0; i < dpc; ++i)
          rot_u (ldi[i]) += loc_rot(i);
      }

    prec_vel_mass.solve (rot_u);
  }
}


int main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace Step35;
      RunTimeParameters::Data_Storage data;
      data.read_data ("parameter-file.prm");

      Utilities::MPI::MPI_InitFinalize mpi_initialization
      (argc, argv, numbers::invalid_unsigned_int);
      {
        deallog.depth_console (data.verbose ? 2 : 0);

        NavierStokesProjection<3> test (data);
        test.run (data.verbose, data.output_interval);
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!"
            << std::endl
            << "Don't forget to brush your teeth :-)"
            << std::endl << std::endl;
  return 0;
}
