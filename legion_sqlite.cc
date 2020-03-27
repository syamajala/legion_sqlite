/* Copyright 2020 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <time.h>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <sqlite3.h>
#include "legion.h"

using namespace Legion;

enum {
  TOP_LEVEL_TASK_ID,
  MODIFY_FIELD_TASK_ID,
  DUMP_FIELDS_TASK_ID
};

enum {
  FID_X,
  FID_Y,
  FID_Z,
};

static sqlite3 *db = NULL;

typedef struct Args
{
  const char *taskname;
  int timestep;
  bool poly;
} Args;

typedef FieldAccessor<READ_ONLY, double, 2> AccessorRO;
typedef FieldAccessor<WRITE_DISCARD, double, 2> AccessorWO;
typedef std::map<FieldID, AccessorRO> ReadAccs;

void check_return(int rc, const char *statement)
{
  if ( rc != SQLITE_OK )
  {
    std::cout << statement << " FAILED! " << rc << std::endl;
  }
}

void modify_field_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const AccessorWO acc(regions[0], fid);

  Rect<2> rect = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (PointInRectIterator<2> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}

void dump_fields_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  const Rect<2> bounds = runtime->get_index_space_domain(
                          task->regions[0].region.get_index_space());
  // UNCOMMENT THIS IF YOUR TASK IS BEING INDEX SPACE LAUNCHED
  // const Point<2> point = task->index_point;
  const Point<2> point(0, 0);

  const Args *args = (const Args*)(task->args);
  const char *taskname = args->taskname;
  const int timestep =  args->timestep;
  const bool poly = args->poly;

  LogicalRegion region(regions[0].get_logical_region());
  FieldSpace fspace = region.get_field_space();

  std::stringstream create;
  create << "CREATE TABLE IF NOT EXISTS " << taskname << " (" << std::endl;
  create << "TIMESTEP int NOT NULL, " << std::endl;
  create << "SUBREGION_X int NOT NULL," << std::endl;
  create << "SUBREGION_Y int NOT NULL," << std::endl;
  create << "IDX_X int NOT NULL," << std::endl;
  create << "IDX_Y int NOT NULL," << std::endl;

  std::stringstream insert;

  insert << "INSERT INTO " << taskname
         << " VALUES (@TIMESTEP, @SUBREGION_X, @SUBREGION_Y, @IDX_X, @IDX_Y, ";

  std::map<FieldID, AccessorRO> field_accs;

  if(!poly)
  {
    std::set<FieldID>::iterator it = (task->regions[0].privilege_fields.begin());
    while(it != task->regions[0].privilege_fields.end())
    {
      FieldID fid = *it;
      const AccessorRO acc(regions[0], fid);
      field_accs.insert(std::pair<FieldID, AccessorRO>(fid, acc));

      const char *field_name;
      runtime->retrieve_name(fspace, fid, field_name);
      create << field_name << " double NOT NULL";
      insert << "@" << field_name;

      ++it;

      if(it != (task->regions[0].privilege_fields.end()))
      {
        create << ",";
        insert << ",";
      }

      create << std::endl;
      insert << " ";
    }
  }
  else
  {
    create << "FIELD text NOT NULL," << std::endl;
    create << "VAL double NOT NULL" << std::endl;

    insert << "@FIELD, @VAL";
    std::set<FieldID>::iterator it = (task->regions[0].privilege_fields.begin());
    while(it != task->regions[0].privilege_fields.end())
    {
      FieldID fid = *it;
      const AccessorRO acc(regions[0], fid);
      field_accs.insert(std::pair<FieldID, AccessorRO>(fid, acc));
      ++it;
    }
  }

  create << ");";
  insert << ");";

  char *errMsg = 0;
  int rc = sqlite3_exec(db, create.str().c_str(), NULL, 0, &errMsg);
  if (rc)
  {
    fprintf(stderr, "Create table %s failed: %d\n", taskname, rc);
  }

  sqlite3_stmt *insert_stmt;
  rc = sqlite3_prepare_v2(db, insert.str().c_str(), -1, &insert_stmt, NULL);

  sqlite3_bind_int(insert_stmt, 1, timestep);
  sqlite3_bind_int(insert_stmt, 2, point[0]);
  sqlite3_bind_int(insert_stmt, 3, point[1]);


  clock_t cStartClock = clock();
  rc = sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, &errMsg);
  int n = 0;

  if(!poly)
  {
    for (long long y = bounds.lo[1]; y <= bounds.hi[1]; y++)
    {
      sqlite3_bind_int(insert_stmt, 5, y);
      for (long long x = bounds.lo[0]; x <= bounds.hi[0]; x++)
      {
        sqlite3_bind_int(insert_stmt, 4, x);

        int field = 6;
        for(std::map<FieldID, AccessorRO>::iterator it = field_accs.begin(); it != field_accs.end(); ++it)
        {
          sqlite3_bind_double(insert_stmt, field, (it->second)[x][y]);
          field++;
        }
        n++;
        sqlite3_step(insert_stmt);
        sqlite3_reset(insert_stmt);
      }
    }
  }
  else
  {
    for (long long y = bounds.lo[1]; y <= bounds.hi[1]; y++)
    {
      sqlite3_bind_int(insert_stmt, 5, y);
      for (long long x = bounds.lo[0]; x <= bounds.hi[0]; x++)
      {
        sqlite3_bind_int(insert_stmt, 4, x);

        for(std::map<FieldID, AccessorRO>::iterator it = field_accs.begin(); it != field_accs.end(); ++it)
        {
          const char *field_name;
          runtime->retrieve_name(fspace, it->first, field_name);

          sqlite3_bind_text(insert_stmt, 6, field_name, -1, SQLITE_STATIC);
          n++;
          sqlite3_bind_double(insert_stmt, 7, (it->second)[x][y]);
          sqlite3_step(insert_stmt);
          sqlite3_reset(insert_stmt);
        }
      }
    }
  }
  sqlite3_finalize(insert_stmt);
  rc = sqlite3_exec(db, "END TRANSACTION", NULL, NULL, &errMsg);
  printf("Imported %d records in %4.2f seconds\n", n, (clock() - cStartClock) / (double)CLOCKS_PER_SEC);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // Create a dummy region and fill it
  IndexSpace ispace = runtime->create_index_space(ctx, Domain(Rect<2>(Point<2>(0, 0), Point<2>(19, 19))));
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(sizeof(double), FID_X);
    runtime->attach_name(fspace, FID_X, "FID_X");
    falloc.allocate_field(sizeof(double), FID_Y);
    runtime->attach_name(fspace, FID_Y, "FID_Y");
    falloc.allocate_field(sizeof(double), FID_Z);
    runtime->attach_name(fspace, FID_Z, "FID_Z");
  }
  LogicalRegion region = runtime->create_logical_region(ctx, ispace, fspace);

  runtime->fill_field<double>(ctx, region, region, FID_X, 0);
  runtime->fill_field<double>(ctx, region, region, FID_Y, 0);
  runtime->fill_field<double>(ctx, region, region, FID_Z, 0);

  int rc;
  rc = sqlite3_open("test.db", &db);
  if ( rc )
  {
    fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
    return;
  } else {
    fprintf(stdout, "Opened database successfully\n");
  }

  char *sErrMsg = 0;
  sqlite3_exec(db, "PRAGMA synchronous = OFF", NULL, NULL, &sErrMsg);
  sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", NULL, NULL, &sErrMsg);
  sqlite3_exec(db, "PRAGMA cache_size=10000", NULL, NULL, &sErrMsg);

  Args arg;
  arg.poly = false;

  for(int timestep = 0; timestep < 10; timestep++)
  {
    arg.timestep = timestep;
    {
      TaskLauncher modify_launcher(MODIFY_FIELD_TASK_ID, TaskArgument(NULL, 0));
      modify_launcher.add_region_requirement(
        RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      modify_launcher.add_field(0/*idx*/, FID_X);
      runtime->execute_task(ctx, modify_launcher);
    }

    {
      arg.taskname = "ModifyField";
      TaskLauncher dump_launcher(DUMP_FIELDS_TASK_ID, TaskArgument(&arg, sizeof(Args)));
      dump_launcher.add_region_requirement(
        RegionRequirement(region, READ_ONLY, EXCLUSIVE, region));
      dump_launcher.add_field(0/*idx*/, FID_X);
      runtime->execute_task(ctx, dump_launcher);
    }
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(MODIFY_FIELD_TASK_ID, "modify_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<modify_field_task>(registrar, "modify_field");
  }

  {
    TaskVariantRegistrar registrar(DUMP_FIELDS_TASK_ID, "dump_fields");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_fields_task>(registrar, "dump_fields");
  }


  return Runtime::start(argc, argv);
}
