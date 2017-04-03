define cc10 void @vfcomp_oge(i64* %baseReg,
                             i64* %sp,
                             i64* %hp,
                             <4 x float> %r1, <4 x float> %r2)
{
  %1 = getelementptr inbounds i64, i64* %sp, i64 0
  %2 = load i64, i64* %1, align 8
  %r = fcmp oge <4 x float> %r1, %r2
  %s = zext <4 x i1> %r to <4 x i32>
  %cont = inttoptr i64 %2 to void (i64*,
                                   i64*,
                                   i64*,
                                   <4 x i32>)*
  tail call cc10 void %cont(i64* %baseReg,
                            i64* %sp,
                            i64* %hp,
                            <4 x i32> %s) noreturn
  ret void
}

define cc10 void @vfselect(i64* %baseReg,
                           i64* %sp,
                           i64* %hp,
                           <4 x i32> %r1, <4 x float> %r2, <4 x float> %r3)
{
  %1 = getelementptr inbounds i64, i64* %sp, i64 0
  %2 = load i64, i64* %1, align 8
  %c = trunc <4 x i32> %r1 to <4 x i1>
  %rr = select <4 x i1> %c, <4 x float> %r2, <4 x float> %r3
  %cont = inttoptr i64 %2 to void (i64*,
                                   i64*,
                                   i64*,
                                   <4 x float>)*
  tail call cc10 void %cont(i64* %baseReg,
                            i64* %sp,
                            i64* %hp,
                            <4 x float> %rr) noreturn
  ret void
}

define cc10 void @vdcomp_oge(i64* %baseReg,
                             i64* %sp,
                             i64* %hp,
                             <2 x double> %r1, <2 x double> %r2)
{
  %1 = getelementptr inbounds i64, i64* %sp, i64 0
  %2 = load i64, i64* %1, align 8
  %r = fcmp oge <2 x double> %r1, %r2
  %s = zext <2 x i1> %r to <2 x i64>
  %cont = inttoptr i64 %2 to void (i64*,
                                   i64*,
                                   i64*,
                                   <2 x i64>)*
  tail call cc10 void %cont(i64* %baseReg,
                            i64* %sp,
                            i64* %hp,
                            <2 x i64> %s) noreturn
  ret void
}

define cc10 void @vdselect(i64* %baseReg,
                           i64* %sp,
                           i64* %hp,
                           <2 x i64> %r1, <2 x double> %r2, <2 x double> %r3)
{
  %1 = getelementptr inbounds i64, i64* %sp, i64 0
  %2 = load i64, i64* %1, align 8
  %c = trunc <2 x i64> %r1 to <2 x i1>
  %rr = select <2 x i1> %c, <2 x double> %r2, <2 x double> %r3
  %cont = inttoptr i64 %2 to void (i64*,
                                   i64*,
                                   i64*,
                                   <2 x double>)*
  tail call cc10 void %cont(i64* %baseReg,
                            i64* %sp,
                            i64* %hp,
                            <2 x double> %rr) noreturn
  ret void
}
